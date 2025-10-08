"""
Helper utilities for notebooks to avoid JSON corruption.

Import these functions inside notebooks and call them after the main pipeline.
"""

from __future__ import annotations

from typing import Dict, Tuple, Iterable
import numpy as np
import pandas as pd


def inject_series_into_df(df: pd.DataFrame, series: np.ndarray, name: str, suffix: str) -> None:
    col = f"{name}{suffix}"
    n, m = len(df), len(series)
    if m == n:
        df[col] = series
    elif m < n:
        tail = series[-1] if m > 0 else 0
        df[col] = np.concatenate([series, np.full(n - m, tail)])
    else:
        df[col] = series[:n]


def calculate_dynamic_baseline_daily(daily_df: pd.DataFrame, periods_df: pd.DataFrame, T_days: int) -> Tuple[np.ndarray, str]:
    """
    Dynamic MA baseline by T:
    - T <= 7:  13-week MA (weekly), scaled by T/7
    - 8..15:   6 bi-weekly MA, scaled by T/14
    - >15:     3 monthly MA, scaled by T/30
    Returns (preds, baseline_name).
    """
    df = daily_df.sort_values('date').copy()
    if T_days <= 7:
        agg = df.resample('W-SUN', on='date')['demand'].sum().reset_index()
        agg.columns = ['period_start', 'agg_demand']
        ma_window, period_ratio, name = 13, T_days/7, '13周移动平均'
    elif T_days <= 15:
        agg = df.resample('2W-SUN', on='date')['demand'].sum().reset_index()
        agg.columns = ['period_start', 'agg_demand']
        ma_window, period_ratio, name = 6, T_days/14, '6期移动平均(双周)'
    else:
        agg = df.resample('MS', on='date')['demand'].sum().reset_index()
        agg.columns = ['period_start', 'agg_demand']
        ma_window, period_ratio, name = 3, T_days/30, '3期移动平均(月)'

    agg['ma'] = agg['agg_demand'].rolling(window=ma_window, min_periods=1).mean()
    agg['ma_forecast'] = agg['ma'].shift(1)

    preds = []
    for i in range(len(periods_df)):
        ref_date = pd.to_datetime(periods_df.iloc[i]['period_start'])
        hist = agg[agg['period_start'] <= ref_date]
        row = hist.iloc[-1] if len(hist) > 0 else agg.iloc[0]
        val = float(row['ma_forecast']) * period_ratio if pd.notna(row['ma_forecast']) else None
        if val is None or np.isnan(val):
            val = preds[-1] if preds else float(row['agg_demand']) * period_ratio
        preds.append(val)
    return np.asarray(preds, dtype=float), name


def ensure_dynamic_baseline_multi(model_predictions: Dict[str, np.ndarray],
                                  model_performance: Dict[str, Dict[str, float]],
                                  merged_df: pd.DataFrame,
                                  test_cleaned: pd.DataFrame,
                                  config,
                                  y_test: Iterable[float],
                                  comparison_df: pd.DataFrame,
                                  calculate_adjusted_metrics_for_overlapping_windows) -> str:
    """For multi-day notebook: compute dynamic baseline via calculate_baseline_ma and inject into outputs.
    Returns the baseline name.
    """
    ma_preds, baseline_name = calculate_baseline_ma(merged_df, test_cleaned, config)
    # Remove 13-week if inappropriate
    if '13周移动平均' in model_predictions and baseline_name != '13周移动平均':
        model_predictions.pop('13周移动平均', None)
        model_performance.pop('13周移动平均', None)
        old_col = '13周移动平均_forecast'
        if old_col in comparison_df.columns:
            comparison_df.drop(columns=[old_col], inplace=True)

    model_predictions[baseline_name] = ma_preds
    y = np.asarray(list(y_test), dtype=float)
    perf = calculate_adjusted_metrics_for_overlapping_windows(y[:len(ma_preds)], ma_preds[:len(y)], config)
    model_performance[baseline_name] = perf
    inject_series_into_df(comparison_df, ma_preds, baseline_name, '_forecast')
    return baseline_name


def ensure_stacking_visible(model_predictions: Dict[str, np.ndarray], comparison_df: pd.DataFrame) -> bool:
    name = 'Stacking'
    if name in model_predictions and f'{name}_forecast' not in comparison_df.columns:
        inject_series_into_df(comparison_df, np.asarray(model_predictions[name], dtype=float), name, '_forecast')
        return True
    return False


from __future__ import annotations

from typing import Iterable, List, Set, Tuple
import numpy as np
import pandas as pd


EXCLUDE_FEATURE_COLUMNS: Set[str] = {
    'date', 'period_start', 'period_end', 'holiday_name'
}


def ensure_datetime(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out


def add_calendar_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    d = ensure_datetime(df, date_col)
    out = d.copy()
    dt = out[date_col]
    out['year'] = dt.dt.year
    out['month'] = dt.dt.month
    out['day'] = dt.dt.day
    out['week_of_year'] = dt.dt.isocalendar().week.astype(int)
    out['day_of_year'] = dt.dt.dayofyear
    out['quarter'] = dt.dt.quarter
    out['is_month_start'] = (dt.dt.is_month_start).astype(int)
    out['is_month_end'] = (dt.dt.is_month_end).astype(int)
    out['day_of_week'] = dt.dt.dayofweek
    out['day_of_week_num'] = out['day_of_week']
    out['is_weekend'] = out['day_of_week'].isin([5, 6]).astype(int)

    # One-hot weekday flags expected by downstream utils
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for i, name in enumerate(weekday_names):
        out[f'is_{name}'] = (out['day_of_week'] == i).astype(int)

    return out


def infer_selected_features(df: pd.DataFrame, include_demand: bool = True) -> List[str]:
    cols: List[str] = []
    if include_demand and 'demand' in df.columns:
        cols.append('demand')
    for c in df.columns:
        if c in EXCLUDE_FEATURE_COLUMNS:
            continue
        if c == 'demand' and include_demand:
            continue
        # Keep numeric or is_* binaries
        s = df[c]
        if s.dtype.kind in {'i', 'u', 'f'}:
            cols.append(c)
        elif c.startswith('is_'):
            cols.append(c)
    # Deduplicate while preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def add_lag_features(df: pd.DataFrame, value_col: str = 'demand', lags: Iterable[int] = (1, 7, 14, 28), date_col: str = 'date') -> pd.DataFrame:
    out = ensure_datetime(df, date_col).copy()
    out = out.sort_values(date_col)
    for l in lags:
        out[f'{value_col}_lag{int(l)}'] = out[value_col].shift(int(l))
    return out


def add_rolling_features(
    df: pd.DataFrame,
    value_col: str = 'demand',
    windows: Iterable[int] = (7, 14, 28),
    date_col: str = 'date',
) -> pd.DataFrame:
    out = ensure_datetime(df, date_col).copy()
    out = out.sort_values(date_col)
    for w in windows:
        out[f'{value_col}_ma{int(w)}'] = out[value_col].rolling(int(w), min_periods=1).mean()
        out[f'{value_col}_std{int(w)}'] = out[value_col].rolling(int(w), min_periods=1).std().fillna(0.0)
    return out


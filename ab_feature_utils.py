import re
from datetime import timedelta
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


A_TIME_FEATURES = {
    'year', 'month', 'day', 'week_of_year', 'day_of_year', 'quarter',
    'is_month_start', 'is_month_end', 'is_weekend', 'day_of_week', 'day_of_week_num',
    # one-hot weekday flags
    'is_Monday', 'is_Tuesday', 'is_Wednesday', 'is_Thursday', 'is_Friday', 'is_Saturday', 'is_Sunday'
}
A_HOLIDAY_FEATURES = {
    'is_holiday', 'is_pre_holiday', 'is_post_holiday'
}
A_INDUSTRY_FEATURES = {
    'product_seasonal_index'
}


def separate_a_b_features(all_features: List[str]) -> Tuple[Set[str], Set[str]]:
    """Split features into A-class (ahead-known) and B-class (needs prediction) sets.
    A-class includes: industry-specific (#19), holiday, basic time, and extra time features.
    Everything else is B-class.
    """
    a_features: Set[str] = set()
    b_features: Set[str] = set()

    for f in all_features:
        # Skip non-feature columns defensively
        if f in {'date', 'holiday_name'}:
            continue

        if (
            f in A_TIME_FEATURES
            or f in A_HOLIDAY_FEATURES
            or f in A_INDUSTRY_FEATURES
        ):
            a_features.add(f)
            continue

        # Aggregated A-style future features (appear in multi-agg dataset)
        if f.startswith('future_'):
            a_features.add(f)
            continue

        # Many time flags start with 'is_' but not all are allowed; keep only weekday/month flags above
        # Any other feature defaults to B-class
        b_features.add(f)

    return a_features, b_features


def _parse_lag_days(feature_name: str) -> Optional[int]:
    m = re.search(r'_lag(\d+)$', feature_name)
    if m:
        return int(m.group(1))
    return None


def _parse_ma_window(feature_name: str) -> Optional[int]:
    m = re.search(r'_ma(\d+)$', feature_name)
    if m:
        return int(m.group(1))
    return None


class BFeatureFillerDaily:
    """Fill B-class features at test time without using test-period ground truth.

    - demand_lagN, demand_maN are computed from known train demand and predicted test demand.
    - Other B features are filled via train-only weekday means as a reasonable proxy.
    """
    def __init__(
        self,
        train_df: pd.DataFrame,
        features: List[str],
        a_features: Set[str],
        b_features: Set[str],
        date_col: str = 'date',
    ) -> None:
        self.date_col = date_col
        self.features = list(features)
        self.a_features = set(a_features)
        self.b_features = set(b_features)

        self.train_df = train_df.copy()
        self.train_df[self.date_col] = pd.to_datetime(self.train_df[self.date_col])
        if 'weekday' not in self.train_df.columns:
            self.train_df['weekday'] = self.train_df[self.date_col].dt.dayofweek

        # Train-only weekday means for all B features (including derived ones as a fallback)
        self.weekday_means_by_feature: Dict[str, Dict[int, float]] = {}
        for f in self.b_features:
            if f in self.train_df.columns:
                grp = self.train_df.groupby('weekday')[f].mean(numeric_only=True)
                self.weekday_means_by_feature[f] = grp.to_dict()

        # Demand history from train for lag/ma backfill
        self.train_demand_series = (
            self.train_df.sort_values(self.date_col)['demand'].astype(float).tolist()
            if 'demand' in self.train_df.columns else []
        )
        self.predicted_test_demands: List[float] = []

    def _get_weekday_fill(self, feature: str, weekday: int, global_mean: float = 0.0) -> float:
        mapping = self.weekday_means_by_feature.get(feature)
        if mapping is None:
            return global_mean
        return float(mapping.get(weekday, global_mean))

    def _get_demand_lag_value(self, lag_days: int) -> float:
        # Combine train demand (older) + predicted (newer) as history
        hist = self.train_demand_series + self.predicted_test_demands
        if len(hist) >= lag_days:
            return float(hist[-lag_days])
        # Not enough history; fallback to earliest available
        return float(hist[0]) if hist else 0.0

    def _get_demand_ma_value(self, window: int) -> float:
        hist = self.train_demand_series + self.predicted_test_demands
        if not hist:
            return 0.0
        take = min(window, len(hist))
        return float(np.mean(hist[-take:]))

    def build_features_for_row(self, row: pd.Series) -> Dict[str, float]:
        """Produce a feature dict for the predictor for this date row.
        A features are copied from the row; B features are filled.
        """
        out: Dict[str, float] = {}
        date_val = pd.to_datetime(row[self.date_col])
        weekday = int(date_val.dayofweek)

        # Pre-compute global means for fallback
        global_means = self.train_df.mean(numeric_only=True).to_dict()

        for f in self.features:
            if f in self.a_features:
                val = row.get(f, 0)
                out[f] = 0.0 if pd.isna(val) or np.isinf(val) else float(val)
                continue

            # B-class handling
            if f.startswith('demand_lag'):
                lag = _parse_lag_days(f)
                out[f] = self._get_demand_lag_value(lag) if lag else 0.0
            elif f.startswith('demand_ma'):
                win = _parse_ma_window(f)
                out[f] = self._get_demand_ma_value(win) if win else 0.0
            else:
                # Other B features via weekday mean (train-only)
                gm = float(global_means.get(f, 0.0))
                out[f] = self._get_weekday_fill(f, weekday, gm)

        return out

    def update_with_prediction(self, pred_demand: float) -> None:
        self.predicted_test_demands.append(float(max(0.0, pred_demand)))


def compute_weekday_demand_pattern(train_df: pd.DataFrame, date_col: str = 'date') -> Dict[int, float]:
    df = train_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['weekday'] = df[date_col].dt.dayofweek
    if 'demand' not in df.columns or df['demand'].sum() <= 0:
        # Fallback to uniform distribution
        return {i: 1.0 / 7.0 for i in range(7)}
    means = df.groupby('weekday')['demand'].mean()
    total = means.sum()
    if total <= 0:
        return {i: 1.0 / 7.0 for i in range(7)}
    weights = (means / total).to_dict()
    # Normalize to sum 1
    z = sum(weights.values())
    return {k: float(v / z) for k, v in weights.items()}


def distribute_aggregate_to_daily(
    total_demand: float,
    dates: List[pd.Timestamp],
    weekday_pattern: Dict[int, float],
) -> List[float]:
    if total_demand <= 0 or not dates:
        return [0.0] * len(dates)
    # Build weights for the dates using weekday pattern
    weights = np.array([weekday_pattern.get(int(d.dayofweek), 1.0) for d in dates], dtype=float)
    s = weights.sum()
    if s <= 0:
        weights = np.ones(len(dates), dtype=float) / len(dates)
    else:
        weights = weights / s
    daily = weights * float(total_demand)
    return daily.tolist()


def build_filled_past_data(
    daily_df: pd.DataFrame,
    past_start: pd.Timestamp,
    past_end: pd.Timestamp,
    a_features: Set[str],
    b_features: Set[str],
    train_df: pd.DataFrame,
    predicted_daily_demand: Dict[pd.Timestamp, float],
    date_col: str = 'date',
) -> pd.DataFrame:
    """Return a copy of daily rows in [past_start, past_end] with B features replaced
    by train-only proxies and predicted demand.
    """
    df = daily_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    mask = (df[date_col] >= past_start) & (df[date_col] <= past_end)
    segment = df.loc[mask].copy()

    # Ensure weekday column
    if 'weekday' not in train_df.columns:
        train_df = train_df.copy()
        train_df[date_col] = pd.to_datetime(train_df[date_col])
        train_df['weekday'] = train_df[date_col].dt.dayofweek
    weekday_global_means = train_df.mean(numeric_only=True).to_dict()
    weekday_means_by_feature = {
        f: train_df.groupby('weekday')[f].mean(numeric_only=True).to_dict()
        for f in b_features if f in train_df.columns
    }

    # Fill B features
    for idx, row in segment.iterrows():
        d = pd.to_datetime(row[date_col])
        weekday = int(d.dayofweek)
        # 'demand' base series
        if 'demand' in segment.columns:
            if d in predicted_daily_demand:
                segment.at[idx, 'demand'] = float(predicted_daily_demand[d])
            # else keep original (likely train side)

        for f in b_features:
            if f not in segment.columns:
                continue
            if f == 'demand':
                continue
            # Use weekday mean from train
            gm = float(weekday_global_means.get(f, 0.0))
            mapping = weekday_means_by_feature.get(f)
            val = float(mapping.get(weekday, gm)) if mapping is not None else gm
            segment.at[idx, f] = val

    return segment


def build_window_features_from_past(
    past_data: pd.DataFrame,
    future_data: pd.DataFrame,
    selected_features: List[str],
) -> Dict[str, float]:
    """Mimic aggregate_to_periods_sliding feature construction but with provided past_data.
    Only creates keys that start with 'past_' and 'future_' derived ones; ignores raw columns.
    """
    agg_row: Dict[str, float] = {
        'period_start': pd.to_datetime(future_data['date']).min(),
        'period_end': pd.to_datetime(future_data['date']).max(),
        'demand': float(future_data['demand'].sum()) if 'demand' in future_data.columns else 0.0,
    }

    # Build past_* features
    for feature in selected_features:
        if feature == 'demand':
            # We will build past_demand_* from the base 'demand' column
            numeric = True
            series = past_data['demand'] if 'demand' in past_data.columns else None
        else:
            if feature not in past_data.columns:
                continue
            series = past_data[feature]
            numeric = series.dtype.kind in {'i', 'u', 'f'}

        if series is None:
            continue

        if numeric:
            try:
                agg_row[f'past_{feature}_mean'] = float(np.nanmean(series.values))
            except Exception:
                agg_row[f'past_{feature}_mean'] = float(series.fillna(0).mean())
            try:
                agg_row[f'past_{feature}_last'] = float(series.iloc[-1])
            except Exception:
                agg_row[f'past_{feature}_last'] = float(series.fillna(0).iloc[-1])
        else:
            mode_val = series.mode()
            agg_row[f'past_{feature}_mode'] = float(mode_val.iloc[0]) if len(mode_val) > 0 else float(series.iloc[-1])

    # Future A-class aggregates
    if 'is_holiday' in future_data.columns:
        agg_row['future_holiday_days'] = int(future_data['is_holiday'].sum())
        agg_row['future_holiday_ratio'] = float(future_data['is_holiday'].mean())
    if 'is_weekend' in future_data.columns:
        agg_row['future_weekend_days'] = int(future_data['is_weekend'].sum())

    weekday_cols = ['is_Monday', 'is_Tuesday', 'is_Wednesday', 'is_Thursday', 'is_Friday', 'is_Saturday', 'is_Sunday']
    for col in weekday_cols:
        if col in future_data.columns:
            agg_row[f'future_{col}_count'] = int(future_data[col].sum())

    return agg_row


def aggregate_prediction_recursive_bsafe(
    daily_df: pd.DataFrame,
    selected_features: List[str],
    product_name: str,
    config,
    trained_predictors: Dict[str, object],
    test_start_date: str,
    verbose: bool = True,
) -> Dict[str, object]:
    """Recursive multi-period prediction using only A-class future info and predicted B-class past info.

    - daily_df: full daily dataframe with base columns including 'date', 'demand', time/holiday flags, and selected_features.
    - selected_features: daily-level features used to construct aggregated past_* features.
    - trained_predictors: mapping of model name -> predictor object having predict_period_aggregate(features_dict).
    - Returns dict with 'predictions' (per model list), 'actuals' (list), 'period_infos' (list of dicts).
    """
    df = daily_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    test_start_date = pd.to_datetime(test_start_date)
    train_df = df[df['date'] < test_start_date].copy()
    test_df = df[df['date'] >= test_start_date].copy()

    past_window = getattr(config, 'past_window', 7)
    future_window = getattr(config, 'future_window', 7)
    step = getattr(config, 'step', 1)

    a_features, b_features = separate_a_b_features(selected_features)

    # Prepare weekday pattern for distributing aggregate predictions into daily
    weekday_pattern = compute_weekday_demand_pattern(train_df, date_col='date')

    # Predicted daily demand dictionary for test dates as we go
    predicted_daily_demand: Dict[pd.Timestamp, float] = {}

    all_predictions: Dict[str, List[float]] = {name: [] for name in trained_predictors.keys()}
    actual_demands: List[float] = []
    period_infos: List[Dict[str, pd.Timestamp]] = []

    current_date = test_start_date

    while current_date <= test_df['date'].max():
        period_start = current_date
        period_end = min(current_date + timedelta(days=future_window - 1), test_df['date'].max())

        # Actual demand for evaluation
        actual_window = test_df[(test_df['date'] >= period_start) & (test_df['date'] <= period_end)]
        actual_demand = float(actual_window['demand'].sum()) if len(actual_window) > 0 else 0.0

        # Build past window dates and filled data
        past_start = period_start - timedelta(days=past_window)
        past_end = period_start - timedelta(days=1)
        past_filled = build_filled_past_data(
            daily_df=df,
            past_start=past_start,
            past_end=past_end,
            a_features=a_features,
            b_features=b_features.union({'demand'}),
            train_df=train_df,
            predicted_daily_demand=predicted_daily_demand,
            date_col='date',
        )

        # Future data slice (A-class features only from calendar)
        future_slice = df[(df['date'] >= period_start) & (df['date'] <= period_end)].copy()

        window_features = build_window_features_from_past(
            past_data=past_filled,
            future_data=future_slice,
            selected_features=selected_features,
        )

        # Predict with all predictors
        for model_name, predictor in trained_predictors.items():
            try:
                pred = float(predictor.predict_period_aggregate(window_features))
            except Exception:
                pred = 0.0
            all_predictions[model_name].append(max(0.0, pred))

        # Choose a reference prediction to distribute daily (use XGBoost if available, else mean)
        if 'XGBoost' in all_predictions:
            ref_pred = all_predictions['XGBoost'][-1]
        else:
            vals = [v[-1] for v in all_predictions.values() if len(v) > 0]
            ref_pred = float(np.mean(vals)) if vals else 0.0

        # Distribute to daily for future use
        window_dates = [period_start + timedelta(days=i) for i in range((period_end - period_start).days + 1)]
        daily_vals = distribute_aggregate_to_daily(ref_pred, window_dates, weekday_pattern)
        for d, v in zip(window_dates, daily_vals):
            predicted_daily_demand[pd.to_datetime(d)] = float(v)

        actual_demands.append(actual_demand)
        period_infos.append({'start': period_start, 'end': period_end})

        # Next window
        current_date = current_date + timedelta(days=step)

    return {
        'predictions': all_predictions,
        'actuals': actual_demands,
        'period_infos': period_infos,
    }

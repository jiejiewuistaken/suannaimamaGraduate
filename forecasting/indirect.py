from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import ab_feature_utils as ab
from . import features as F


@dataclass
class IndirectConfig:
    past_window: int = 28
    future_window: int = 7
    step: int = 7
    date_col: str = 'date'


class _Regressor:
    def __init__(self) -> None:
        self._name = 'GBR'
        self._model = None  # type: ignore

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
            model = GradientBoostingRegressor(random_state=17)
            model.fit(X, y)
            self._model = model
            self._name = 'GBR'
            return
        except Exception:
            pass

        # Fallback to ridge
        self._name = 'Ridge'
        reg = 1e-3
        XTX = X.T @ X + reg * np.eye(X.shape[1])
        XTy = X.T @ y
        try:
            w = np.linalg.solve(XTX, XTy)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(XTX) @ XTy
        self._model = ('ridge', w)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self._model, 'predict'):
            return self._model.predict(X)  # type: ignore
        w = self._model[1]
        return X @ w

    @property
    def name(self) -> str:
        return self._name


def _build_training_matrix(daily_df: pd.DataFrame, selected_features: List[str], cfg: IndirectConfig, train_end_date: str) -> Tuple[np.ndarray, np.ndarray]:
    df = F.add_calendar_features(daily_df, date_col=cfg.date_col)
    df = F.add_lag_features(df, value_col='demand', lags=(1, 7, 14, 28), date_col=cfg.date_col)
    df = F.add_rolling_features(df, value_col='demand', windows=(7, 14, 28), date_col=cfg.date_col)
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    df = df.sort_values(cfg.date_col)

    # Training subset
    train_end = pd.to_datetime(train_end_date)
    train_df = df[df[cfg.date_col] <= train_end].copy()

    # Drop rows with NaN after lags/rolling
    train_df = train_df.dropna()

    # Build matrix
    X_cols = [c for c in selected_features if c in train_df.columns and c != 'demand']
    if 'demand' not in train_df.columns:
        raise ValueError('Dataframe must contain demand column')
    X = train_df[X_cols].astype(float).values
    y = train_df['demand'].astype(float).values
    return X, y


def _aggregate_windows(df: pd.DataFrame, preds_by_date: Dict[pd.Timestamp, float], cfg: IndirectConfig, test_start_date: str) -> Dict[str, object]:
    d = df.copy()
    d[cfg.date_col] = pd.to_datetime(d[cfg.date_col])
    d = d.sort_values(cfg.date_col)
    test_start = pd.to_datetime(test_start_date)
    test_df = d[d[cfg.date_col] >= test_start]
    if len(test_df) == 0:
        return {'predictions': {'Indirect': []}, 'actuals': [], 'period_infos': []}

    all_predictions: List[float] = []
    actual_demands: List[float] = []
    period_infos: List[Dict[str, pd.Timestamp]] = []

    current = test_start
    last_date = test_df[cfg.date_col].max()
    while current <= last_date:
        period_start = current
        period_end = min(current + pd.Timedelta(days=cfg.future_window - 1), last_date)
        window_dates = [period_start + pd.Timedelta(days=i) for i in range((period_end - period_start).days + 1)]

        # Aggregate predictions and actuals
        pred_sum = float(sum(max(0.0, float(preds_by_date.get(pd.to_datetime(dt), 0.0))) for dt in window_dates))
        actual_sum = float(test_df[(test_df[cfg.date_col] >= period_start) & (test_df[cfg.date_col] <= period_end)]['demand'].sum())

        all_predictions.append(pred_sum)
        actual_demands.append(actual_sum)
        period_infos.append({'start': period_start, 'end': period_end})

        current = current + pd.Timedelta(days=cfg.step)

    return {'predictions': {'Indirect': all_predictions}, 'actuals': actual_demands, 'period_infos': period_infos}


class IndirectDailyPipeline:
    def __init__(self, cfg: Optional[IndirectConfig] = None) -> None:
        self.cfg = cfg or IndirectConfig()
        self.model = _Regressor()
        self.selected_features: Optional[List[str]] = None

    def fit(self, daily_df: pd.DataFrame, selected_features: Optional[List[str]], train_end_date: str) -> None:
        base = F.add_calendar_features(daily_df, date_col=self.cfg.date_col)
        base = F.add_lag_features(base, value_col='demand', lags=(1, 7, 14, 28), date_col=self.cfg.date_col)
        base = F.add_rolling_features(base, value_col='demand', windows=(7, 14, 28), date_col=self.cfg.date_col)
        if selected_features is None:
            selected_features = F.infer_selected_features(base)
        self.selected_features = list(selected_features)
        X, y = _build_training_matrix(base, self.selected_features, self.cfg, train_end_date)
        self.model.fit(X, y)

    def rolling_origin_backtest(self, daily_df: pd.DataFrame, test_start_date: str) -> Dict[str, object]:
        # Prepare dataframe with calendar features
        df = F.add_calendar_features(daily_df, date_col=self.cfg.date_col)
        df[self.cfg.date_col] = pd.to_datetime(df[self.cfg.date_col])
        df = df.sort_values(self.cfg.date_col)

        test_start = pd.to_datetime(test_start_date)
        train_df = df[df[self.cfg.date_col] < test_start].copy()
        test_df = df[df[self.cfg.date_col] >= test_start].copy()
        if self.selected_features is None:
            self.selected_features = F.infer_selected_features(df)

        a_features, b_features = ab.separate_a_b_features(self.selected_features)
        filler = ab.BFeatureFillerDaily(
            train_df=train_df,
            features=self.selected_features,
            a_features=a_features,
            b_features=b_features,
            date_col=self.cfg.date_col,
        )

        # Predict day-by-day recursively
        preds_by_date: Dict[pd.Timestamp, float] = {}
        for _, row in test_df.iterrows():
            feats = filler.build_features_for_row(row)
            # Build X order aligned to selected_features except demand target
            X_vec = np.array([feats.get(c, 0.0) for c in self.selected_features if c != 'demand'], dtype=float).reshape(1, -1)
            yhat = float(self.model.predict(X_vec)[0])
            yhat = max(0.0, yhat)
            d = pd.to_datetime(row[self.cfg.date_col])
            preds_by_date[d] = yhat
            filler.update_with_prediction(yhat)

        # Aggregate to windows for comparison with direct
        return _aggregate_windows(df, preds_by_date, self.cfg, test_start_date)


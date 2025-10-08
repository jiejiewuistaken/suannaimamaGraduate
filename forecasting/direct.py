from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import ab_feature_utils as ab
from . import features as F


@dataclass
class DirectConfig:
    past_window: int = 28
    future_window: int = 7
    step: int = 7
    date_col: str = 'date'


class _Vectorizer:
    def __init__(self) -> None:
        self.feature_names: List[str] = []

    def fit(self, rows: List[Dict[str, float]]) -> None:
        names: List[str] = sorted(set(k for r in rows for k in r.keys()))
        self.feature_names = names

    def transform(self, rows: List[Dict[str, float]]) -> np.ndarray:
        if not self.feature_names:
            return np.zeros((len(rows), 0), dtype=float)
        X = np.zeros((len(rows), len(self.feature_names)), dtype=float)
        name_to_idx = {n: i for i, n in enumerate(self.feature_names)}
        for i, r in enumerate(rows):
            for k, v in r.items():
                j = name_to_idx.get(k)
                if j is not None:
                    X[i, j] = float(v)
        return X


class _Regressor:
    def __init__(self) -> None:
        self._vec = _Vectorizer()
        self._model = None  # type: ignore
        self._name = 'Linear'

    def fit(self, X_rows: List[Dict[str, float]], y: List[float]) -> None:
        self._vec.fit(X_rows)
        X = self._vec.transform(X_rows)
        yt = np.asarray(y, dtype=float)

        # Try sklearn first
        model = None
        try:
            from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
            model = GradientBoostingRegressor(random_state=17)
            model.fit(X, yt)
            self._model = model
            self._name = 'GBR'
            return
        except Exception:
            pass

        # Fallback: ridge closed-form
        self._name = 'Ridge'
        reg = 1e-3
        XTX = X.T @ X + reg * np.eye(X.shape[1])
        XTy = X.T @ yt
        try:
            w = np.linalg.solve(XTX, XTy)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(XTX) @ XTy
        self._model = ('ridge', w)

    def predict(self, X_rows: List[Dict[str, float]]) -> np.ndarray:
        X = self._vec.transform(X_rows)
        if hasattr(self._model, 'predict'):
            return self._model.predict(X)  # type: ignore
        # ridge
        w = self._model[1]
        return X @ w

    # Interface expected by ab.aggregate_prediction_recursive_bsafe
    def predict_period_aggregate(self, features_dict: Dict[str, float]) -> float:
        pred = float(self.predict([features_dict])[0])
        return max(0.0, pred)

    @property
    def name(self) -> str:
        return self._name


def _iter_training_anchors(df: pd.DataFrame, cfg: DirectConfig, train_end: pd.Timestamp) -> List[pd.Timestamp]:
    start_date = df[cfg.date_col].min() + pd.Timedelta(days=cfg.past_window)
    last_start = train_end - pd.Timedelta(days=cfg.future_window - 1)
    if last_start < start_date:
        return []
    anchors: List[pd.Timestamp] = []
    cur = start_date
    while cur <= last_start:
        anchors.append(cur)
        cur = cur + pd.Timedelta(days=cfg.step)
    return anchors


def build_training_dataset(
    daily_df: pd.DataFrame,
    selected_features: List[str],
    cfg: DirectConfig,
    train_end_date: str,
) -> Tuple[List[Dict[str, float]], List[float]]:
    df = F.add_calendar_features(daily_df, date_col=cfg.date_col)
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    df = df.sort_values(cfg.date_col).reset_index(drop=True)

    train_end = pd.to_datetime(train_end_date)
    anchors = _iter_training_anchors(df, cfg, train_end)

    X_rows: List[Dict[str, float]] = []
    y_vals: List[float] = []

    for start in anchors:
        past_start = start - pd.Timedelta(days=cfg.past_window)
        past_end = start - pd.Timedelta(days=1)
        future_end = start + pd.Timedelta(days=cfg.future_window - 1)

        past_slice = df[(df[cfg.date_col] >= past_start) & (df[cfg.date_col] <= past_end)]
        future_slice = df[(df[cfg.date_col] >= start) & (df[cfg.date_col] <= future_end)]

        if len(future_slice) <= 0 or len(past_slice) <= 0:
            continue

        features_dict = ab.build_window_features_from_past(
            past_data=past_slice,
            future_data=future_slice,
            selected_features=selected_features,
        )
        # Drop non-numeric/meta keys from features
        for k in ['period_start', 'period_end', 'demand']:
            if k in features_dict:
                features_dict.pop(k, None)
        X_rows.append(features_dict)
        y_vals.append(float(future_slice['demand'].sum()))

    return X_rows, y_vals


class DirectAggregatePipeline:
    def __init__(self, cfg: Optional[DirectConfig] = None) -> None:
        self.cfg = cfg or DirectConfig()
        self.model = _Regressor()
        self.selected_features: Optional[List[str]] = None

    def fit(self, daily_df: pd.DataFrame, selected_features: Optional[List[str]], train_end_date: str) -> None:
        if selected_features is None:
            selected_features = F.infer_selected_features(daily_df)
        self.selected_features = list(selected_features)
        X_rows, y_vals = build_training_dataset(daily_df, self.selected_features, self.cfg, train_end_date)
        self.model.fit(X_rows, y_vals)

    def rolling_origin_backtest(self, daily_df: pd.DataFrame, test_start_date: str) -> Dict[str, object]:
        if self.selected_features is None:
            self.selected_features = F.infer_selected_features(daily_df)
        result = ab.aggregate_prediction_recursive_bsafe(
            daily_df=daily_df,
            selected_features=self.selected_features,
            product_name='default',
            config=self.cfg,
            trained_predictors={self.model.name: self.model},
            test_start_date=test_start_date,
            verbose=False,
        )
        return result


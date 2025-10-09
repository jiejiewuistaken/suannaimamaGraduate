from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

Metric = Literal["MAE", "RMSE", "R2", "MAPE", "WMAPE"]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true.astype(float)
    yp = y_pred.astype(float)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    mask = np.abs(y_true) > eps
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def wmape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.sum(np.abs(y_true))
    if denom <= eps:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


@dataclass
class EvalResult:
    mae: float
    rmse: float
    r2: float
    mape: float
    wmape: float

    def to_dict(self) -> Dict[str, float]:
        return {"MAE": self.mae, "RMSE": self.rmse, "R2": self.r2, "MAPE": self.mape, "WMAPE": self.wmape}


def eval_array(y_true: Iterable[float], y_pred: Iterable[float]) -> EvalResult:
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    return EvalResult(
        mae=mae(yt, yp),
        rmse=rmse(yt, yp),
        r2=r2(yt, yp),
        mape=mape(yt, yp),
        wmape=wmape(yt, yp),
    )


# Aggregations

def aggregate_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: Literal["W-MON", "M"] = "W-MON",
    agg: Literal["sum", "mean"] = "sum",
) -> pd.DataFrame:
    g = df[[date_col, value_col]].copy()
    g[date_col] = pd.to_datetime(g[date_col])
    grouped = g.resample(freq, on=date_col)[value_col]
    if agg == "sum":
        out = grouped.sum().reset_index()
    else:
        out = grouped.mean().reset_index()
    return out


def align_on_period(
    actual_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    date_col: str,
    actual_col: str,
    pred_col: str,
    freq: Literal["W-MON", "M"] = "W-MON",
    agg: Literal["sum", "mean"] = "sum",
) -> pd.DataFrame:
    a = aggregate_series(actual_df, date_col, actual_col, freq=freq, agg=agg)
    p = aggregate_series(pred_df, date_col, pred_col, freq=freq, agg=agg)
    merged = a.merge(p, on=date_col, how="inner", suffixes=("_actual", "_pred"))
    merged.columns = [date_col, "actual", "pred"]
    return merged


def evaluate_period(
    actual_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    date_col: str = "date",
    actual_col: str = "actual",
    pred_col: str = "pred",
    freq: Literal["W-MON", "M"] = "W-MON",
    agg: Literal["sum", "mean"] = "sum",
) -> Tuple[EvalResult, pd.DataFrame]:
    aligned = align_on_period(actual_df.rename(columns={actual_col: "actual"}),
                              pred_df.rename(columns={pred_col: "pred"}),
                              date_col=date_col, actual_col="actual", pred_col="pred", freq=freq, agg=agg)
    res = eval_array(aligned["actual"].values, aligned["pred"].values)
    return res, aligned


# Baselines

def baseline_ma(df: pd.DataFrame, date_col: str, value_col: str, window: int = 13, freq: str = "W-MON") -> pd.DataFrame:
    s = aggregate_series(df, date_col, value_col, freq=freq, agg="sum")
    s["ma"] = s[value_col].rolling(window=window, min_periods=1).mean().shift(1)
    s.rename(columns={value_col: "actual"}, inplace=True)
    s["pred"] = s["ma"].fillna(method="ffill").fillna(0.0)
    return s[[date_col, "actual", "pred"]]


def evaluate_with_baselines(
    actual_daily: pd.DataFrame,
    pred_daily: pd.DataFrame,
    date_col: str = "date",
    actual_col: str = "demand",
    pred_col: str = "pred",
    freq: str = "W-MON",
    agg: str = "sum",
    linear_baseline: Optional[pd.DataFrame] = None,
    ma_window: int = 13,
) -> Dict[str, Tuple[EvalResult, pd.DataFrame]]:
    results: Dict[str, Tuple[EvalResult, pd.DataFrame]] = {}

    # primary model
    res, aligned = evaluate_period(
        actual_df=actual_daily.rename(columns={actual_col: "actual"}),
        pred_df=pred_daily.rename(columns={pred_col: "pred"}),
        date_col=date_col, freq=freq, agg=agg,
    )
    results["model"] = (res, aligned)

    # 13-week MA
    ma_df = baseline_ma(actual_daily.rename(columns={actual_col: "actual"}), date_col=date_col, value_col="actual", window=ma_window, freq=freq)
    res_ma = eval_array(ma_df["actual"].values, ma_df["pred"].values)
    results[f"MA_{ma_window}"] = (res_ma, ma_df)

    # optional linear regression baseline, expects daily preds already provided
    if linear_baseline is not None:
        lin_res, lin_aligned = evaluate_period(
            actual_df=actual_daily.rename(columns={actual_col: "actual"}),
            pred_df=linear_baseline.rename(columns={pred_col: "pred"}),
            date_col=date_col, freq=freq, agg=agg,
        )
        results["LinearRegression"] = (lin_res, lin_aligned)

    return results

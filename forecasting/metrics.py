from __future__ import annotations

from typing import Iterable, List, Tuple
import numpy as np


def _to_numpy(a: Iterable[float]) -> np.ndarray:
    if isinstance(a, np.ndarray):
        return a.astype(float)
    return np.array(list(a), dtype=float)


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    if yt.size == 0:
        return 0.0
    return float(np.mean(np.abs(yp - yt)))


def mape_safe(y_true: Iterable[float], y_pred: Iterable[float], eps: float = 1e-8) -> float:
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    denom = np.maximum(np.abs(yt), eps)
    if yt.size == 0:
        return 0.0
    return float(np.mean(np.abs((yp - yt) / denom)))


def smape(y_true: Iterable[float], y_pred: Iterable[float], eps: float = 1e-8) -> float:
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    denom = np.maximum((np.abs(yt) + np.abs(yp)) / 2.0, eps)
    if yt.size == 0:
        return 0.0
    return float(np.mean(np.abs(yp - yt) / denom))


def wape(y_true: Iterable[float], y_pred: Iterable[float], eps: float = 1e-8) -> float:
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    denom = np.maximum(np.sum(np.abs(yt)), eps)
    if yt.size == 0:
        return 0.0
    return float(np.sum(np.abs(yp - yt)) / denom)


def evaluate_aggregate(y_true: Iterable[float], y_pred: Iterable[float]) -> dict:
    return {
        'MAE': mae(y_true, y_pred),
        'MAPE': mape_safe(y_true, y_pred),
        'sMAPE': smape(y_true, y_pred),
        'WAPE': wape(y_true, y_pred),
    }


def summarize(name_and_series: List[Tuple[str, Iterable[float], Iterable[float]]]) -> List[Tuple[str, dict]]:
    out: List[Tuple[str, dict]] = []
    for name, yt, yp in name_and_series:
        out.append((name, evaluate_aggregate(yt, yp)))
    return out


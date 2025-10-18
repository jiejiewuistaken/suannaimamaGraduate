"""
Shapley-Lipschitz robust feature-based newsvendor implementation.

Implements the two-step solution from:
- Zhang, Luhao; Yang, Jincheng; Gao, Rui (2024). Optimal Robust Policy for Feature-Based Newsvendor. Management Science 70(4):2315–2329.

Step I (in-sample LP, Proposition 2):
  min  (max(b, h) * rho) * L  +  (1/n) * sum_{k=1..K} sum_{i=1..n_k} psi_{k,i}
  s.t.  |y_j - y_k| <= L * ||x_j - x_k||,  for all j,k in [K]
        psi_{k,i} >= h * (y_k - z_{k,i})
        psi_{k,i} >= b * (z_{k,i} - y_k)
        L >= beta >= 0

Step II (Shapley/Lipschitz extension at a new feature x):
  y(x) = argmin_y max_k |y - y_k| / ||x - x_k||
  Equivalent LP: minimize t s.t. |y - y_k| <= t * ||x - x_k||, t >= 0.

This module exposes a high-level API:
- ShapleyRobustFeatureNewsvendor.fit(X, z): fits in-sample robust policy {y_k} for distinct feature points {x_k}
- .predict(X_new): returns robust orders for new features via Shapley/Lipschitz extension

Notes:
- Distances are computed in a configurable norm (default Euclidean). Features may optionally be standardized before distance computation.
- Grouping of observations to distinct feature points defaults to exact equality on the provided (possibly preprocessed) feature vectors; optional rounding tolerance can be applied to reduce K.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import numpy as np

try:
    import pulp as pl
except Exception as e:  # pragma: no cover
    pl = None  # Will raise at runtime in fit/predict with a helpful message


ArrayLike = np.ndarray


def _ensure_pulp_available() -> None:
    if pl is None:
        raise RuntimeError(
            "pulp is not available. Please install with `pip install pulp` and retry."
        )


def _pairwise_distances(X: ArrayLike, norm: str = "l2") -> ArrayLike:
    """Compute pairwise distances between rows of X.

    Supports:
    - 'l2' (Euclidean)
    - 'l1' (Manhattan)
    - 'linf' (Chebyshev)
    """
    X = np.asarray(X, dtype=float)
    K = X.shape[0]
    D = np.zeros((K, K), dtype=float)
    if norm == "l2":
        # Efficient vectorized computation
        # D_{ij} = ||x_i - x_j||_2
        # Using (x - y)^2 = x^2 + y^2 - 2 x.y
        sq = np.sum(X * X, axis=1, keepdims=True)
        sq_mat = sq + sq.T - 2.0 * (X @ X.T)
        np.maximum(sq_mat, 0.0, out=sq_mat)
        D = np.sqrt(sq_mat)
    elif norm == "l1":
        for i in range(K):
            D[i, :] = np.sum(np.abs(X[i] - X), axis=1)
    elif norm == "linf":
        for i in range(K):
            D[i, :] = np.max(np.abs(X[i] - X), axis=1)
    else:
        raise ValueError(f"Unsupported norm: {norm}")
    np.fill_diagonal(D, 0.0)
    return D


def _group_by_features(
    X: ArrayLike,
    z: ArrayLike,
    rounding_eps: Optional[float] = None,
) -> Tuple[ArrayLike, List[List[float]], List[int]]:
    """Group observations by distinct feature vectors (optionally rounded).

    Returns:
    - X_groups: (K, d) array of distinct feature points in the order of appearance
    - Z_groups: list of lists, demands per group k
    - group_index: length-n list mapping each observation to its group index
    """
    X = np.asarray(X, dtype=float)
    z = np.asarray(z, dtype=float).reshape(-1)
    if X.shape[0] != z.shape[0]:
        raise ValueError("X and z must have the same number of rows")

    if rounding_eps is not None and rounding_eps > 0:
        X_key = np.round(X / rounding_eps) * rounding_eps
    else:
        X_key = X.copy()

    key_to_index: Dict[Tuple[float, ...], int] = {}
    X_groups_list: List[List[float]] = []
    Z_groups: List[List[float]] = []
    group_index: List[int] = []

    for i in range(X_key.shape[0]):
        key = tuple(X_key[i].tolist())
        if key not in key_to_index:
            key_to_index[key] = len(X_groups_list)
            X_groups_list.append(list(X[i].tolist()))  # store original X, not rounded
            Z_groups.append([])
        k = key_to_index[key]
        Z_groups[k].append(float(z[i]))
        group_index.append(k)

    X_groups = np.array(X_groups_list, dtype=float)
    return X_groups, Z_groups, group_index


@dataclass
class ShapleyRobustFeatureNewsvendorConfig:
    h_cost: float
    b_cost: float
    rho: float
    beta: float = 1.0  # scaling lower bound for L (>= 0)
    norm: str = "l2"  # 'l2'|'l1'|'linf'
    rounding_eps: Optional[float] = None  # set to e.g. 0.0 or small >0 to control K
    time_limit_sec: Optional[int] = 180
    solver_msg: bool = False


class ShapleyRobustFeatureNewsvendor:
    """End-to-end robust feature-based newsvendor.

    After fit(), the policy is represented by (X_groups, y_star) and extended to new
    x via the Lipschitz/Shapley extension by solving a tiny LP per query.
    """

    def __init__(self, config: ShapleyRobustFeatureNewsvendorConfig):
        self.config = config
        self._is_fitted: bool = False
        self.X_groups: Optional[ArrayLike] = None
        self.Z_groups: Optional[List[List[float]]] = None
        self.y_star: Optional[ArrayLike] = None
        self.L_star: Optional[float] = None
        self._pairwise_D: Optional[ArrayLike] = None

    def fit(self, X: ArrayLike, z: ArrayLike) -> "ShapleyRobustFeatureNewsvendor":
        _ensure_pulp_available()
        cfg = self.config

        X_groups, Z_groups, _ = _group_by_features(
            X=X, z=z, rounding_eps=cfg.rounding_eps
        )
        K = X_groups.shape[0]
        N = int(sum(len(g) for g in Z_groups))
        if K == 0:
            raise ValueError("No data provided")
        if N == 0:
            raise ValueError("No demand observations provided")

        D = _pairwise_distances(X_groups, norm=cfg.norm)
        self._pairwise_D = D

        # LP variables
        mdl = pl.LpProblem("InSampleRobustFeatureNewsvendor", pl.LpMinimize)
        y_vars: Dict[int, pl.LpVariable] = {
            k: pl.LpVariable(f"y_{k}", lowBound=None, upBound=None, cat=pl.LpContinuous)
            for k in range(K)
        }
        L_var = pl.LpVariable("L", lowBound=cfg.beta, upBound=None, cat=pl.LpContinuous)

        # psi variables per observation
        psi_vars: List[pl.LpVariable] = []
        obs_index: List[Tuple[int, int]] = []  # (k, i) index of psi
        for k, z_list in enumerate(Z_groups):
            for i, _ in enumerate(z_list):
                v = pl.LpVariable(f"psi_{k}_{i}", lowBound=0.0, upBound=None, cat=pl.LpContinuous)
                psi_vars.append(v)
                obs_index.append((k, i))

        # Pairwise Lipschitz constraints
        for j in range(K):
            for k in range(K):
                if j == k:
                    continue
                dist_jk = float(D[j, k])
                if dist_jk == 0.0:
                    # Identical features after rounding -> force equality of y
                    mdl += y_vars[j] - y_vars[k] == 0.0
                else:
                    mdl += y_vars[j] - y_vars[k] <= L_var * dist_jk
                    mdl += y_vars[k] - y_vars[j] <= L_var * dist_jk

        # Newsvendor piecewise linear cost upper envelope per observation
        for idx, (k, i) in enumerate(obs_index):
            z_val = float(Z_groups[k][i])
            psi = psi_vars[idx]
            mdl += psi >= cfg.h_cost * (y_vars[k] - z_val)
            mdl += psi >= cfg.b_cost * (z_val - y_vars[k])

        coeff_front = max(cfg.h_cost, cfg.b_cost) * cfg.rho
        objective = coeff_front * L_var + (1.0 / N) * pl.lpSum(psi_vars)
        mdl += objective

        solver = pl.PULP_CBC_CMD(msg=cfg.solver_msg, timeLimit=cfg.time_limit_sec)
        res = mdl.solve(solver)
        status = pl.LpStatus[res]
        if status not in ("Optimal", "Not Solved", "Infeasible", "Unbounded", "Undefined"):
            # Fallback label variants; but we still guard for infeasible/unbounded below
            pass
        if status != "Optimal":
            raise RuntimeError(f"LP did not solve to optimality. Status={status}")

        y_star = np.array([pl.value(y_vars[k]) for k in range(K)], dtype=float)
        L_star = float(pl.value(L_var))

        self.X_groups = X_groups
        self.Z_groups = Z_groups
        self.y_star = y_star
        self.L_star = L_star
        self._is_fitted = True
        return self

    def _predict_single_via_extension(self, x: ArrayLike) -> float:
        _ensure_pulp_available()
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted")
        assert self.X_groups is not None and self.y_star is not None

        Xg = self.X_groups
        yg = self.y_star
        # Quick check for exact match
        # Use allclose to be robust to tiny numeric differences
        for k in range(Xg.shape[0]):
            if np.allclose(x, Xg[k], rtol=0.0, atol=1e-12):
                return float(yg[k])

        # Build tiny LP: minimize t s.t. |y - y_k| <= t * ||x - x_k||
        # Precompute distances
        diff = Xg - np.asarray(x, dtype=float).reshape(1, -1)
        if self.config.norm == "l2":
            dist = np.sqrt(np.maximum(np.sum(diff * diff, axis=1), 0.0))
        elif self.config.norm == "l1":
            dist = np.sum(np.abs(diff), axis=1)
        elif self.config.norm == "linf":
            dist = np.max(np.abs(diff), axis=1)
        else:
            raise ValueError(f"Unsupported norm: {self.config.norm}")

        mdl = pl.LpProblem("ShapleyExtensionAtX", pl.LpMinimize)
        y_var = pl.LpVariable("y_ext", lowBound=None, upBound=None, cat=pl.LpContinuous)
        t_var = pl.LpVariable("t", lowBound=0.0, upBound=None, cat=pl.LpContinuous)
        for k in range(Xg.shape[0]):
            d = float(dist[k])
            if d == 0.0:
                # Must interpolate exactly; force equality
                mdl += y_var == float(yg[k])
            else:
                mdl += y_var - float(yg[k]) <= t_var * d
                mdl += float(yg[k]) - y_var <= t_var * d
        mdl += t_var
        solver = pl.PULP_CBC_CMD(msg=self.config.solver_msg, timeLimit=30)
        res = mdl.solve(solver)
        status = pl.LpStatus[res]
        if status != "Optimal":
            raise RuntimeError(f"Extension LP not optimal. Status={status}")
        return float(pl.value(y_var))

    def predict(self, X_new: ArrayLike) -> ArrayLike:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted")
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            return np.array([self._predict_single_via_extension(X_new)], dtype=float)
        preds = np.zeros((X_new.shape[0],), dtype=float)
        for i in range(X_new.shape[0]):
            preds[i] = self._predict_single_via_extension(X_new[i])
        return preds

    # Convenience helpers
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def get_in_sample_policy(self) -> Tuple[ArrayLike, ArrayLike, float]:
        """Return (X_groups, y_star, L_star)."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted")
        assert self.X_groups is not None and self.y_star is not None and self.L_star is not None
        return self.X_groups.copy(), self.y_star.copy(), float(self.L_star)

    def score_expected_cost(self, X: ArrayLike, z: ArrayLike) -> float:
        """Compute empirical expected newsvendor cost under the learned policy.

        Note: For inputs outside the training feature set, we use the extension.
        """
        X = np.asarray(X, dtype=float)
        z = np.asarray(z, dtype=float).reshape(-1)
        if X.shape[0] != z.shape[0]:
            raise ValueError("X and z must have the same number of rows")
        y_pred = self.predict(X)
        h = self.config.h_cost
        b = self.config.b_cost
        over = np.maximum(y_pred - z, 0.0)
        under = np.maximum(z - y_pred, 0.0)
        return float(np.mean(h * over + b * under))

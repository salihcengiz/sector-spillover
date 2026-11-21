"""Doğrusal kıyas modelleri (train örneklerinde en küçük kareler).

- HAROwn    : her sektör yalnızca kendi HAR regresörleriyle (sektör başına ayrı OLS).
- HARMarket : HAROwn + piyasa göstergesi. Piyasa göstergesi, HAR regresörlerinin 49 sektörün
              tamamı üzerinden alınan ortalamasıdır (sektör gruplaması değildir).
- RidgeAll  : her sektör, tüm sektörlerin HAR regresörleriyle (N·K girdi). Ceza katsayısı
              val örneklerinde seçilir; katsayılar yalnızca train'de öğrenilir.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..data.panel import Panel
from ..prep.splits import Splits
from .base import Baseline, har_regressors, require_finite


def _with_intercept(Z: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones(Z.shape[:-1] + (1,)), Z], axis=-1)


class _PerSectorOLS(Baseline):
    """Her sektör için ayrı OLS: y[t, j] ~ 1 + design[t, j, :]."""

    def __init__(self, windows: Optional[Sequence[int]] = None):
        self.windows = windows
        self.coef_: Optional[np.ndarray] = None   # (N, 1 + P)

    def design(self, panel: Panel) -> np.ndarray:  # (T, N, P)
        raise NotImplementedError

    def fit(self, panel: Panel, splits: Splits) -> "_PerSectorOLS":
        D = _with_intercept(self.design(panel))
        idx = splits.train
        require_finite(D, idx, f"{self.name} regresörleri")
        self.coef_ = np.stack([np.linalg.lstsq(D[idx, j], panel.y[idx, j], rcond=None)[0]
                               for j in range(panel.N)])
        return self

    def predict(self, panel: Panel) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Önce fit() çağrılmalı.")
        return np.einsum("tnp,np->tn", _with_intercept(self.design(panel)), self.coef_)


class HAROwn(_PerSectorOLS):
    name = "har_own"

    def design(self, panel: Panel) -> np.ndarray:
        return har_regressors(panel, self.windows)


class HARMarket(_PerSectorOLS):
    name = "har_market"

    def design(self, panel: Panel) -> np.ndarray:
        own = har_regressors(panel, self.windows)                  # (T, N, K)
        market = own.mean(axis=1, keepdims=True)                   # (T, 1, K)
        return np.concatenate([own, np.broadcast_to(market, own.shape)], axis=-1)


class RidgeAll(Baseline):
    """y[t, :] ~ tüm sektörlerin HAR regresörleri. Girdiler train istatistikleriyle standartlaştırılır;
    sabit terim cezalandırılmaz. alpha, n_train ile ölçeklenir (ceza = alpha · n · ||β||²).
    """

    name = "ridge_all"
    DEFAULT_ALPHAS = tuple(np.logspace(-4, 2, 13))

    def __init__(self, windows: Optional[Sequence[int]] = None, alphas: Sequence[float] = DEFAULT_ALPHAS):
        self.windows = windows
        self.alphas = tuple(float(a) for a in alphas)
        self.alpha_: Optional[float] = None
        self.val_mse_: Optional[dict] = None

    def _features(self, panel: Panel) -> np.ndarray:
        Z = har_regressors(panel, self.windows)
        return Z.reshape(panel.T, -1)                               # (T, N·K)

    def _solve(self, Zs: np.ndarray, Yc: np.ndarray, alpha: float) -> np.ndarray:
        n, p = Zs.shape
        return np.linalg.solve(Zs.T @ Zs + alpha * n * np.eye(p), Zs.T @ Yc)

    def fit(self, panel: Panel, splits: Splits) -> "RidgeAll":
        Z = self._features(panel)
        tr, va = splits.train, splits.val
        require_finite(Z, tr, "ridge_all regresörleri")
        require_finite(Z, va, "ridge_all regresörleri")
        self.z_mean_, self.z_std_ = Z[tr].mean(axis=0), np.maximum(Z[tr].std(axis=0), 1e-12)
        self.y_mean_ = panel.y[tr].mean(axis=0)
        Zs = (Z - self.z_mean_) / self.z_std_
        Yc = panel.y[tr] - self.y_mean_

        self.val_mse_ = {}
        best = None
        for alpha in self.alphas:
            beta = self._solve(Zs[tr], Yc, alpha)
            mse = float(np.mean((Zs[va] @ beta + self.y_mean_ - panel.y[va]) ** 2))
            self.val_mse_[alpha] = mse
            if best is None or mse < best[0]:
                best = (mse, alpha, beta)
        _, self.alpha_, self.beta_ = best
        return self

    def predict(self, panel: Panel) -> np.ndarray:
        if self.alpha_ is None:
            raise RuntimeError("Önce fit() çağrılmalı.")
        Zs = (self._features(panel) - self.z_mean_) / self.z_std_
        return Zs @ self.beta_ + self.y_mean_

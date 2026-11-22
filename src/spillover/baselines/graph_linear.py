"""GHAR: doğrusal graf modeli (Zhang vd., 2022/2025).

    ŷ_{i,t} = α_i + x_{i,t}·β + (W x_t)_i·γ

- x_{i,t}: düğüm özellikleri. `features="view"` → panelin kendi özellikleri (aylıkta ayın ortalaması, en yüksek ve
  en düşük günü); `features="har"` → HAR değişkenleri (aylıkta 1, 3, 12 ay; makaledeki gibi).
- W: normalize komşuluk matrisi (graphs.normalize_adjacency). Boş grafta model grafsız birleştirilmiş doğrusal
  modele indirgenir.
- α_i sektöre özel; β ve γ bütün sektörlerde ortak. Train örneklerinde OLS ile çözülür (sektör sabit etkileri,
  sektör içi ortalamadan arındırma ile).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..data.panel import Panel
from ..prep.splits import Splits
from .base import har_regressors, require_finite

GHAR_FEATURES = ("view", "har")


class GHAR:
    name = "ghar"

    def __init__(self, features: str = "view"):
        if features not in GHAR_FEATURES:
            raise ValueError(f"Bilinmeyen özellik seti: {features!r}. Geçerli: {GHAR_FEATURES}.")
        self.features = features
        self.coef_: Optional[np.ndarray] = None
        self.alpha_: Optional[np.ndarray] = None
        self.W_: Optional[np.ndarray] = None

    def node_features(self, panel: Panel) -> np.ndarray:
        return panel.X if self.features == "view" else har_regressors(panel)

    def design(self, panel: Panel, W: np.ndarray) -> np.ndarray:
        """(T, N, P): kendi özellikleri, graf boş değilse yanına komşu ortalamaları (W x)."""
        Z = self.node_features(panel)
        if not np.any(W):
            return Z
        return np.concatenate([Z, np.einsum("ij,tjf->tif", W, Z)], axis=-1)

    def fit(self, panel: Panel, splits: Splits, W: np.ndarray) -> "GHAR":
        W = np.asarray(W, dtype=float)
        if W.shape != (panel.N, panel.N):
            raise ValueError(f"W boyutu {W.shape}, beklenen {(panel.N, panel.N)}.")
        D = self.design(panel, W)
        idx = splits.train
        require_finite(D, idx, "GHAR özellikleri")
        Dt, yt = D[idx], panel.y[idx]                              # (n, N, P), (n, N)
        D_mean, y_mean = Dt.mean(axis=0), yt.mean(axis=0)            # sektör ortalamaları
        Dc = (Dt - D_mean).reshape(-1, D.shape[-1])
        yc = (yt - y_mean).reshape(-1)
        self.coef_ = np.linalg.lstsq(Dc, yc, rcond=None)[0]
        self.alpha_ = y_mean - D_mean @ self.coef_
        self.W_ = W
        return self

    def predict(self, panel: Panel) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Önce fit() çağrılmalı.")
        return self.design(panel, self.W_) @ self.coef_ + self.alpha_

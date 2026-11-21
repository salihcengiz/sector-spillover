"""Standartlaştırma: istatistikler yalnızca train bölümünden hesaplanır.

- Girdiler her (sektör, özellik) çifti için ayrı standartlaştırılır ve ±clip aralığına kırpılır
  (1987 gibi uç günlerin modeli doyurmaması için).
- Hedef her sektör için ayrı standartlaştırılır. Değerlendirme öncesinde inverse_y ile geri çevrilir.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..data.panel import Panel
from .splits import Splits

_MIN_STD = 1e-12


@dataclass
class Standardizer:
    clip: Optional[float] = 10.0
    x_mean: Optional[np.ndarray] = None   # (N, F)
    x_std: Optional[np.ndarray] = None    # (N, F)
    y_mean: Optional[np.ndarray] = None   # (N,)
    y_std: Optional[np.ndarray] = None    # (N,)

    def fit(self, panel: Panel, splits: Splits) -> "Standardizer":
        x_train = panel.X[splits.train_rows]
        y_train = panel.y[splits.train]
        self.x_mean = x_train.mean(axis=0)
        self.x_std = np.maximum(x_train.std(axis=0), _MIN_STD)
        self.y_mean = y_train.mean(axis=0)
        self.y_std = np.maximum(y_train.std(axis=0), _MIN_STD)
        return self

    def _check(self):
        if self.x_mean is None:
            raise RuntimeError("Önce fit() çağrılmalı.")

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        self._check()
        Z = (X - self.x_mean) / self.x_std
        if self.clip is not None:
            Z = np.clip(Z, -self.clip, self.clip)
        return Z.astype(np.float32)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        self._check()
        return ((y - self.y_mean) / self.y_std).astype(np.float32)

    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        self._check()
        return np.asarray(y_scaled, dtype=float) * self.y_std + self.y_mean

    def state_dict(self) -> dict:
        self._check()
        return {"clip": self.clip, "x_mean": self.x_mean, "x_std": self.x_std,
                "y_mean": self.y_mean, "y_std": self.y_std}

    @classmethod
    def from_state_dict(cls, state: dict) -> "Standardizer":
        return cls(**state)

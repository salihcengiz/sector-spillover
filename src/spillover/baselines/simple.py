"""Parametre öğrenmeyen kıyas modelleri: persistence ve EWMA."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..data.panel import Panel
from .base import Baseline

# RiskMetrics değerleri: günlük 0.94, aylık 0.97.
EWMA_LAMBDA = {"daily": 0.94, "monthly": 0.97}


class Persistence(Baseline):
    """Gelecek dönemin riski = son dönemin riski (log_rv[t]; günlükte son 5 gün, aylıkta bu ay)."""

    name = "persistence"

    def predict(self, panel: Panel) -> np.ndarray:
        return panel.log_rv.copy()


class EWMA(Baseline):
    """σ²_t = λ σ²_{t-1} + (1-λ) r2_t, tahmin log(σ²_t + eps). Başlangıç σ²_0 = r2_0."""

    name = "ewma"

    def __init__(self, lam: Optional[float] = None):
        if lam is not None and not 0.0 < lam < 1.0:
            raise ValueError("lam 0 ile 1 arasında olmalı.")
        self.lam = lam

    def predict(self, panel: Panel) -> np.ndarray:
        lam = self.lam if self.lam is not None else EWMA_LAMBDA[panel.frequency]
        r2 = panel.r2
        var = np.empty_like(r2)
        var[0] = r2[0]
        for t in range(1, len(r2)):
            var[t] = lam * var[t - 1] + (1.0 - lam) * r2[t]
        return np.log(var + panel.eps)

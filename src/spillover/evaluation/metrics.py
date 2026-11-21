"""Kayıp fonksiyonları ve OOS R².

Hedef ve tahminler log gerçekleşen varyans ölçeğindedir: y = log(RV + eps).
- MSE   : log ölçeğinde kare hata.
- QLIKE : varyans ölçeğinde Patton (2011) kaybı, L = σ²/h − log(σ²/h) − 1, σ² = e^y, h = e^ŷ.
          log ölçeğinde d = y − ŷ ile L = e^d − d − 1 olur. Gürültülü RV vekiline karşı sıralamayı
          korur ve riskin eksik tahminini fazla tahmininden daha ağır cezalandırır.
- OOS R²: 1 − SSE(model) / SSE(train ortalaması), sektör bazında.
"""
from __future__ import annotations

import numpy as np

from ..data.panel import Panel
from ..prep.splits import Splits


def squared_error(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    return (np.asarray(y) - np.asarray(y_hat)) ** 2


def qlike(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    d = np.asarray(y) - np.asarray(y_hat)
    return np.expm1(d) - d


LOSSES = {"mse": squared_error, "qlike": qlike}


def oos_r2(y: np.ndarray, y_hat: np.ndarray, benchmark: np.ndarray) -> np.ndarray:
    """(n, N) hedef ve tahmin, (N,) kıyas (genelde train ortalaması) → (N,) R²."""
    y = np.asarray(y)
    sse = ((y - y_hat) ** 2).sum(axis=0)
    sst = ((y - np.asarray(benchmark)) ** 2).sum(axis=0)
    return 1.0 - sse / sst


def train_mean(panel: Panel, splits: Splits) -> np.ndarray:
    """OOS R²'nin kıyası: her sektörün train hedef ortalaması (N,)."""
    return panel.y[splits.train].mean(axis=0)

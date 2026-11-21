"""Kıyas modellerinin ortak arayüzü ve HAR regresörleri.

Bütün kıyas modelleri Panel üzerinde çalışır ve frekansı bilmez. Tahminler hedefle aynı
ölçektedir (log gerçekleşen varyans; ölçeklenmemiş). Böylece LSTM tahminleri
`Standardizer.inverse_y` ile geri çevrildikten sonra aynı metriklerle karşılaştırılabilir.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np

from ..data.panel import Panel
from ..data.views import trailing_mean
from ..prep.splits import Splits

# HAR pencereleri (dönem cinsinden). Günlük: gün / hafta / ay (Corsi 2009).
# Aylık: ay / çeyrek / yıl (aynı fikrin aylık karşılığı).
HAR_WINDOWS = {"daily": (1, 5, 22), "monthly": (1, 3, 12)}


def har_regressors(panel: Panel, windows: Optional[Sequence[int]] = None) -> np.ndarray:
    """(T, N, K) HAR regresörleri: log(son w dönemin ortalama r²'si + eps), her w için.

    Günlük varsayılanda bunlar log_r2, log_rv5 ve log_rv22 özelliklerinin aynısıdır.
    Yalnızca t ve öncesine bakar; ilk max(w)-1 dönem NaN'dır.
    """
    windows = tuple(windows or HAR_WINDOWS[panel.frequency])
    return np.stack([np.log(trailing_mean(panel.r2, w) + panel.eps) for w in windows], axis=-1)


def require_finite(array: np.ndarray, idx: np.ndarray, what: str) -> None:
    if not np.isfinite(array[idx]).all():
        raise ValueError(f"{what} bazı örneklerde tanımsız; lookback, en uzun HAR penceresinden kısa olabilir.")


class Baseline(ABC):
    """fit(panel, splits) yalnızca train (gerekirse val) örneklerine bakar.
    predict(panel) her t dönemi için (T, N) tahmin döndürür; tanımsız yerlerde NaN.
    """

    name: str = ""

    def fit(self, panel: Panel, splits: Splits) -> "Baseline":
        return self

    @abstractmethod
    def predict(self, panel: Panel) -> np.ndarray:
        ...

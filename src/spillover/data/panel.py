"""Panel: görünümlerin (günlük, aylık…) ürettiği ortak veri yapısı.

Hazırlık (bölme, ölçekleme, pencereleme), modeller ve değerlendirme yalnızca Panel
ile çalışır; verinin hangi frekanstan geldiğini bilmeleri gerekmez.

Zaman indeksi t bir "dönem"dir (günlük görünümde gün, aylık görünümde ay):
- X[t]      : t dönemi sonunda bilinen girdiler (yalnızca t ve öncesine ait bilgi).
- r2[t]     : t dönemindeki günlük kare getirilerin ortalaması (günlükte r_t², aylıkta ayın ortalaması).
              Kıyas modelleri (HAR, EWMA) bu seriden frekansı bilmeden kurulur.
- log_rv[t] : t dönemine ait gerçekleşen risk, log(ortalama r² + eps).
- y[t]      : t döneminden sonraki hedef penceresinin riski. Hedef penceresi
              t+1 … t+horizon dönemlerini kapsar; son `horizon` dönemde y NaN'dır.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Panel:
    dates: pd.DatetimeIndex   # (T,) her dönemin etiket tarihi (o dönemin son işlem günü)
    sectors: tuple            # (N,) veri setindeki sırasıyla sektör kısa adları
    X: np.ndarray             # (T, N, F) girdiler
    feature_names: tuple      # (F,)
    y: np.ndarray             # (T, N) hedef
    r2: np.ndarray            # (T, N) dönemin ortalama günlük kare getirisi (%²)
    log_rv: np.ndarray        # (T, N) dönem riski; hedefle aynı pencere uzunluğunda
    frequency: str            # "daily" | "monthly"
    horizon: int              # hedef penceresinin ileriye uzandığı dönem sayısı (purge uzunluğu)
    eps: float

    def __post_init__(self):
        T, N = len(self.dates), len(self.sectors)
        if self.X.shape != (T, N, len(self.feature_names)):
            raise ValueError(f"X boyutu {self.X.shape}, beklenen {(T, N, len(self.feature_names))}.")
        for name in ("y", "r2", "log_rv"):
            if getattr(self, name).shape != (T, N):
                raise ValueError(f"{name} boyutu {getattr(self, name).shape}, beklenen {(T, N)}.")
        if self.horizon < 1:
            raise ValueError("horizon en az 1 olmalı.")

    @property
    def T(self) -> int:
        return len(self.dates)

    @property
    def N(self) -> int:
        return len(self.sectors)

    @property
    def F(self) -> int:
        return len(self.feature_names)

    def feature(self, name: str) -> np.ndarray:
        """Tek bir özelliği (T, N) dizisi olarak döndürür."""
        return self.X[:, :, self.feature_names.index(name)]

    def first_valid_index(self) -> int:
        """Tüm girdilerin tanımlı olduğu ilk dönem.

        Kayan pencereli özellikler (örn. 22 günlük RV) başta tanımsızdır. Bu dönemden
        sonra eksik değer olmamalıdır; varsa hata verir.
        """
        finite = np.isfinite(self.X).all(axis=(1, 2))
        if not finite.any():
            raise ValueError("Hiçbir dönemde girdilerin tamamı tanımlı değil.")
        first = int(np.argmax(finite))
        if not finite[first:].all():
            bad = self.dates[first + int(np.argmin(finite[first:]))].date()
            raise ValueError(f"{bad} tarihinde girdilerde eksik değer var.")
        return first

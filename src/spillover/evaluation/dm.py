"""Diebold–Mariano testi (Newey–West uzun dönem varyansı, Harvey–Leybourne–Newbold düzeltmesi).

d_t = L_a,t − L_b,t. İstatistik negatifse a modelinin kaybı daha düşüktür (a daha iyi).
h dönem ileriye uzanan hedefler örtüştüğü için d_t en fazla h−1 gecikmeye kadar ilişkili
olabilir; bu yüzden varsayılan Newey–West gecikmesi h−1'dir (günlük 5 gün → 4, aylık → 0).
p-değeri iki yönlü ve standart normaldir (n ≥ 100 için t dağılımıyla farkı ihmal edilebilir).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class DMResult:
    stat: float
    p_value: float
    mean_diff: float   # ortalama(L_a − L_b)
    n: int
    lag: int


def newey_west_lrv(d: np.ndarray, lag: int) -> float:
    """Ortalaması çıkarılmış serinin Bartlett ağırlıklı uzun dönem varyansı (1/n normalizasyonu)."""
    d = np.asarray(d, dtype=float)
    d = d - d.mean()
    n = len(d)
    lrv = float(d @ d) / n
    for k in range(1, min(lag, n - 1) + 1):
        lrv += 2.0 * (1.0 - k / (lag + 1)) * float(d[k:] @ d[:-k]) / n
    return lrv


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int = 1,
                    lag: Optional[int] = None, small_sample: bool = True) -> DMResult:
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    if d.ndim != 1 or not np.isfinite(d).all():
        raise ValueError("Kayıplar tek boyutlu ve sonlu olmalı.")
    n = len(d)
    lag = horizon - 1 if lag is None else int(lag)
    mean_diff = float(d.mean())
    lrv = newey_west_lrv(d, lag)
    if lrv <= 0.0:
        stat = 0.0 if mean_diff == 0.0 else math.copysign(math.inf, mean_diff)
    else:
        stat = mean_diff / math.sqrt(lrv / n)
        if small_sample:
            h = horizon
            stat *= math.sqrt(max(n + 1 - 2 * h + h * (h - 1) / n, 0.0) / n)
    p_value = math.erfc(abs(stat) / math.sqrt(2.0))
    return DMResult(stat=stat, p_value=p_value, mean_diff=mean_diff, n=n, lag=lag)

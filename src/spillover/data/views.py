"""Görünüm katmanı: günlük getirilerden belirli bir frekansta Panel üretir.

Her görünüm aynı arayüzü uygular: `build(returns) -> Panel`. Böylece aynı model,
config'teki `data.frequency` değiştirilerek günlük ya da aylık veriyle beslenebilir.
Yeni bir frekans (örn. haftalık) eklemek için View'dan türeyen bir sınıf yazıp
VIEWS sözlüğüne eklemek yeterlidir.

Tüm getiriler yüzde cinsindendir. Log dönüşümlerinde kullanılan `eps` (%² cinsinden),
veri 2 ondalıkla yuvarlandığı için oluşan tam 0.00 getirilerin log'unu sınırlar.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .panel import Panel

DEFAULT_EPS = 0.01


def trailing_mean(x: np.ndarray, k: int) -> np.ndarray:
    """Zaman ekseni (sondan ikinci eksen) boyunca son k dönemin ortalaması.

    x boyutu (..., T, N). Çıktıda t satırı x[t-k+1 … t] ortalamasıdır; ilk k-1 satır NaN.
    """
    if k < 1:
        raise ValueError("k en az 1 olmalı.")
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    if x.shape[-2] < k:
        return out
    csum = np.cumsum(x, axis=-2)
    out[..., k - 1:, :] = csum[..., k - 1:, :]
    out[..., k:, :] -= csum[..., :-k, :]
    return out / k


# ---------------------------------------------------------------- günlük görünüm

_LOG_RV = re.compile(r"log_rv(\d+)")


def _daily_window(name: str) -> int:
    """Bir günlük özelliğin hesaplanması için gereken gün sayısı."""
    if name in ("ret", "log_r2"):
        return 1
    match = _LOG_RV.fullmatch(name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Bilinmeyen günlük özellik: {name!r}. Geçerli: ret, log_r2, log_rv<k>.")


def daily_features(r: np.ndarray, names, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Günlük getirilerden (..., T, N) → özellikler (..., T, N, F).

    Özellikler yalnızca t ve öncesine bakar:
      ret       : r_t
      log_r2    : log(r_t² + eps)
      log_rv<k> : log(son k günün r² ortalaması + eps), örn. log_rv5, log_rv22
    """
    r = np.asarray(r, dtype=float)
    r2 = r ** 2
    columns = []
    for name in names:
        window = _daily_window(name)
        if name == "ret":
            columns.append(r)
        elif name == "log_r2":
            columns.append(np.log(r2 + eps))
        else:
            columns.append(np.log(trailing_mean(r2, window) + eps))
    return np.stack(columns, axis=-1)


class View(ABC):
    """Günlük getiri tablosunu belirli bir frekansta Panel'e dönüştüren arayüz."""

    frequency: str = ""

    @abstractmethod
    def build(self, returns: pd.DataFrame) -> Panel:
        """returns: satırlar işlem günleri, sütunlar sektörler (yüzde getiri)."""


class DailyView(View):
    """Günlük görünüm: her dönem bir işlem günü.

    Hedef: sonraki `horizon` işlem gününün log ortalama r²'si,
    y[t] = log(mean(r²[t+1 … t+horizon]) + eps).
    """

    frequency = "daily"
    DEFAULT_FEATURES = ("ret", "log_r2", "log_rv5", "log_rv22")

    def __init__(self, features=DEFAULT_FEATURES, horizon: int = 5, eps: float = DEFAULT_EPS):
        self.features = tuple(features)
        self.horizon = int(horizon)
        self.eps = float(eps)
        for name in self.features:
            _daily_window(name)  # bilinmeyen özellik adını erkenden yakala

    @property
    def history(self) -> int:
        """Son satırın özelliklerini hesaplamak için gereken gün sayısı (senaryo analizinde kullanılır)."""
        return max(_daily_window(name) for name in self.features)

    def compute_features(self, r: np.ndarray) -> np.ndarray:
        """(…, T, N) günlük getiri → (…, T, N, F) özellik. Senaryo analizi de bunu çağırır."""
        return daily_features(r, self.features, self.eps)

    def build(self, returns: pd.DataFrame) -> Panel:
        r = returns.to_numpy(dtype=float)
        h = self.horizon
        r2 = r ** 2
        log_rv = np.log(trailing_mean(r2, h) + self.eps)      # t dönemi: son h günün riski
        y = np.full_like(log_rv, np.nan)
        y[:-h] = log_rv[h:]                                   # y[t] = risk(t+1 … t+h)
        return Panel(
            dates=pd.DatetimeIndex(returns.index),
            sectors=tuple(returns.columns),
            X=self.compute_features(r),
            feature_names=self.features,
            y=y,
            r2=r2,
            log_rv=log_rv,
            frequency=self.frequency,
            horizon=h,
            eps=self.eps,
        )


# ----------------------------------------------------------------- aylık görünüm

MONTHLY_FEATURES = ("mean", "max", "min", "std")


class MonthlyView(View):
    """Aylık görünüm (danışman önerisi): her dönem bir takvim ayı.

    Her ay ve her sektör için özellikler, o ayın günlük getirilerinden hesaplanır:
      mean : ayın ortalama günlük getirisi
      max  : ayın en yüksek günlük getirisi
      min  : ayın en düşük günlük getirisi
      std  : ayın günlük getiri standart sapması (isteğe bağlı 4. özellik)
    Hedef: sonraki ayın log ortalama r²'si, y[t] = log_rv[t+1].
    Dönem etiketi, ayın son işlem günüdür.
    """

    frequency = "monthly"
    DEFAULT_FEATURES = ("mean", "max", "min")

    def __init__(self, features=DEFAULT_FEATURES, eps: float = DEFAULT_EPS):
        self.features = tuple(features)
        self.eps = float(eps)
        unknown = [f for f in self.features if f not in MONTHLY_FEATURES]
        if unknown:
            raise ValueError(f"Bilinmeyen aylık özellik: {unknown}. Geçerli: {MONTHLY_FEATURES}.")

    def build(self, returns: pd.DataFrame) -> Panel:
        month = returns.index.to_period("M")
        grouped = returns.groupby(month)
        stats = {}
        for name in self.features:
            if name == "std":
                stats[name] = grouped.std(ddof=1)
            else:
                stats[name] = getattr(grouped, name)()
        X = np.stack([stats[name].to_numpy(dtype=float) for name in self.features], axis=-1)

        r2 = (returns ** 2).groupby(month).mean().to_numpy(dtype=float)
        log_rv = np.log(r2 + self.eps)
        y = np.full_like(log_rv, np.nan)
        y[:-1] = log_rv[1:]

        last_day = returns.index.to_series().groupby(month).max()
        return Panel(
            dates=pd.DatetimeIndex(last_day.to_numpy()),
            sectors=tuple(returns.columns),
            X=X,
            feature_names=self.features,
            y=y,
            r2=r2,
            log_rv=log_rv,
            frequency=self.frequency,
            horizon=1,
            eps=self.eps,
        )


VIEWS = {DailyView.frequency: DailyView, MonthlyView.frequency: MonthlyView}


def build_view(frequency: str, **params) -> View:
    """Config'teki frekans adından görünüm nesnesi üretir: build_view("monthly", features=[...])."""
    if frequency not in VIEWS:
        raise ValueError(f"Bilinmeyen frekans: {frequency!r}. Geçerli: {sorted(VIEWS)}.")
    return VIEWS[frequency](**params)

"""Train / val / test bölmesi (frekanstan bağımsız).

Her bölüm bir veya birkaç tarih aralığından oluşur (`SplitSpec`). Hiçbir aralığa girmeyen
dönemler kullanılmaz. Bir örnek, tahminin yapıldığı t dönemidir:

- Girdi penceresi [t-lookback+1, t]. Önceki dönemlere taşabilir, çünkü bu geçmiş bilgidir.
- Purge: hedef penceresi [t+1, t+horizon] başka bir bölüme taşamaz. Hiçbir bölüme girmeyen
  (kullanılmayan) dönemlere taşıyabilir; model o dönemleri hiç görmediği için bu sızıntı değildir
  ve test pencerelerinin son günleri / ayları kaybolmaz.
- Embargo (E dönem): bir train örneğinin [t-E, t+horizon+E] aralığında hiç val/test dönemi
  bulunmamalıdır. Böylece val/test pencerelerinin iki yanında boşluk kalır. Ölçekleyicinin fit
  edildiği satırlar (train_rows) da [t-E, t] aralığında val/test dönemi içermez.

Aynı yapı bütün düzenleri kapsar:
- Tek uzun bölme (aylık B): SplitSpec.sequential("2010-12-31", "2016-12-31", embargo=1)
- Kriz pencereleri (günlük): her pencere için ayrı bir SplitSpec
- Walk-forward (aylık A): walk_forward_specs(...) her test yılı için bir SplitSpec üretir
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..data.panel import Panel

SPLIT_NAMES = ("train", "val", "test")
UNUSED = -1                                   # hiçbir bölüme girmeyen dönem
_PAST_END = -2                                # verinin sonundan sonrası
DEFAULT_EMBARGO = {"daily": 22, "monthly": 1}  # dönem cinsinden (22 işlem günü ≈ 1 ay)


def _normalize_ranges(ranges, split: str) -> tuple:
    """[[başlangıç, bitiş], ...] → ((Timestamp|None, Timestamp|None), ...). None = açık uç."""
    out = []
    for r in ranges:
        if len(r) != 2:
            raise ValueError(f"{split}: her aralık [başlangıç, bitiş] biçiminde olmalı, gelen: {r!r}.")
        start, end = (None if v is None else pd.Timestamp(v) for v in r)
        if start is not None and end is not None and start > end:
            raise ValueError(f"{split}: başlangıç bitişten sonra olamaz: {r!r}.")
        out.append((start, end))
    if not out:
        raise ValueError(f"{split}: en az bir tarih aralığı gerekli.")
    return tuple(out)


@dataclass(frozen=True)
class SplitSpec:
    """Bölmenin tarih cinsinden tanımı. Aralıkların iki ucu da dahildir."""
    train: tuple
    val: tuple
    test: tuple
    embargo: int = 0
    name: str = ""

    def __post_init__(self):
        for split in SPLIT_NAMES:
            object.__setattr__(self, split, _normalize_ranges(getattr(self, split), split))
        if int(self.embargo) < 0:
            raise ValueError("embargo negatif olamaz.")
        object.__setattr__(self, "embargo", int(self.embargo))

    @classmethod
    def sequential(cls, train_end, val_end, embargo: int = 0, name: str = "sequential") -> "SplitSpec":
        """train ≤ train_end < val ≤ val_end < test (verinin sonuna kadar)."""
        train_end, val_end = pd.Timestamp(train_end), pd.Timestamp(val_end)
        if not train_end < val_end:
            raise ValueError("train_end, val_end'den önce olmalı.")
        day = pd.Timedelta(days=1)
        return cls(train=[(None, train_end)], val=[(train_end + day, val_end)],
                   test=[(val_end + day, None)], embargo=embargo, name=name)

    @classmethod
    def from_config(cls, cfg: dict) -> "SplitSpec":
        """Config sözlüğünden kurar. İki biçim kabul edilir:
            {train_end, val_end, embargo}                  → sequential
            {train: [[a, b], ...], val: [...], test: [...], embargo}
        """
        cfg = dict(cfg)
        if "train_end" in cfg:
            return cls.sequential(cfg.pop("train_end"), cfg.pop("val_end"), **cfg)
        return cls(**cfg)

    def labels(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Her döneme bölüm numarası verir: 0 = train, 1 = val, 2 = test, -1 = kullanılmıyor."""
        dates = pd.DatetimeIndex(dates)
        labels = np.full(len(dates), UNUSED, dtype=np.int64)
        for code, split in enumerate(SPLIT_NAMES):
            for start, end in getattr(self, split):
                mask = np.ones(len(dates), dtype=bool)
                if start is not None:
                    mask &= dates >= start
                if end is not None:
                    mask &= dates <= end
                if (labels[mask] != UNUSED).any():
                    raise ValueError(f"{split} aralığı {start}–{end} başka bir aralıkla çakışıyor.")
                labels[mask] = code
        return labels


def walk_forward_specs(first_test_year: int, last_test_year: int, val_years: int,
                       embargo: int = 0) -> list:
    """Genişleyen pencereli walk-forward: her test yılı Y için bir SplitSpec (adı "y<Y>").

        train : verinin başı → (Y - val_years - 1)-12-31
        val   : (Y - val_years)-01-01 → (Y - 1)-12-31
        test  : Y-01-01 → Y-12-31
    İlk katman, aynı val_years ile kurulan sıralı bölmenin (B) ilk test yılıyla aynıdır.
    """
    if val_years < 1 or last_test_year < first_test_year:
        raise ValueError("val_years ≥ 1 ve last_test_year ≥ first_test_year olmalı.")
    specs = []
    for year in range(int(first_test_year), int(last_test_year) + 1):
        specs.append(SplitSpec(
            train=[(None, f"{year - val_years - 1}-12-31")],
            val=[(f"{year - val_years}-01-01", f"{year - 1}-12-31")],
            test=[(f"{year}-01-01", f"{year}-12-31")],
            embargo=embargo, name=f"y{year}"))
    return specs


@dataclass(frozen=True)
class Splits:
    train: np.ndarray        # örnek indeksleri (tahmin dönemi t)
    val: np.ndarray
    test: np.ndarray
    train_rows: np.ndarray   # ölçekleyicinin fit edileceği satırlar
    lookback: int
    spec: Optional[SplitSpec] = field(default=None, compare=False)

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in SPLIT_NAMES:
            raise KeyError(f"Geçersiz bölüm: {name!r}. Geçerli: {SPLIT_NAMES}.")
        return getattr(self, name)

    def sizes(self) -> dict:
        return {name: len(self[name]) for name in SPLIT_NAMES}


def make_splits(panel: Panel, spec, lookback: int) -> Splits:
    """Panel için örnek indekslerini üretir.

    Args:
        spec: SplitSpec ya da config sözlüğü (bkz. SplitSpec.from_config).
        lookback: Girdi penceresi uzunluğu (dönem cinsinden; örn. 60 gün veya 12 ay).
    """
    if isinstance(spec, dict):
        spec = SplitSpec.from_config(spec)
    if lookback < 1:
        raise ValueError("lookback en az 1 olmalı.")
    labels = spec.labels(panel.dates)
    T, h, E = panel.T, panel.horizon, spec.embargo
    pos = np.arange(T)
    first = panel.first_valid_index()
    usable = (pos - lookback + 1 >= first) & np.isfinite(panel.y).all(axis=1)

    # Purge: t+1 … t+h dönemleri t'nin bölümünde ya da kullanılmayan dönemlerde olmalı.
    padded = np.concatenate([labels, np.full(h, _PAST_END)])
    same_ahead = np.ones(T, dtype=bool)
    for k in range(1, h + 1):
        ahead = padded[k:k + T]
        same_ahead &= (ahead == labels) | (ahead == UNUSED)

    # Embargo: [t+lo, t+hi] aralığında val/test dönemi var mı?
    eval_count = np.concatenate([[0], np.cumsum((labels == 1) | (labels == 2))])

    def touches_eval(lo: int, hi: int) -> np.ndarray:
        a = np.clip(pos + lo, 0, T)
        b = np.clip(pos + hi + 1, 0, T)
        return eval_count[b] - eval_count[a] > 0

    out = {}
    for code, name in enumerate(SPLIT_NAMES):
        in_segment = labels == code
        if not in_segment.any():
            raise ValueError(f"{name} döneminde hiç gözlem yok; tarih aralıklarını kontrol edin.")
        mask = in_segment & usable & same_ahead
        if name == "train":
            mask &= ~touches_eval(-E, h + E)
        idx = pos[mask]
        if idx.size == 0:
            raise ValueError(f"{name} bölümünde kullanılabilir örnek kalmadı (lookback/horizon/embargo çok büyük olabilir).")
        out[name] = idx

    train_rows = pos[(labels == 0) & (pos >= first) & ~touches_eval(-E, 0)]
    return Splits(train_rows=train_rows, lookback=int(lookback), spec=spec, **out)

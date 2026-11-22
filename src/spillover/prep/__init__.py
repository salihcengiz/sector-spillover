"""Hazırlık katmanı: Panel → bölme → ölçekleme → pencereleme (frekanstan bağımsız).

Tipik kullanım:
    panel = MonthlyView().build(returns)        # veya DailyView()
    spec = SplitSpec.sequential("2010-12-31", "2016-12-31", embargo=1)
    data = prepare(panel, spec, lookback=12)
    for xb, yb in data.dataset("train").batches(128, shuffle=True, rng=rng):
        ...
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.panel import Panel
from .scaling import Standardizer
from .splits import DEFAULT_EMBARGO, SPLIT_NAMES, UNUSED, SplitSpec, Splits, make_splits, walk_forward_specs
from .windows import WindowDataset


@dataclass(frozen=True)
class PreparedData:
    panel: Panel
    splits: Splits
    scaler: Standardizer
    X: np.ndarray   # (T, N, F) ölçeklenmiş girdiler (float32)
    y: np.ndarray   # (T, N) ölçeklenmiş hedef (float32; tanımsız dönemlerde NaN)

    def dataset(self, split: str) -> WindowDataset:
        return WindowDataset(self.X, self.y, self.splits[split], self.splits.lookback)


def prepare(panel: Panel, spec, lookback: int, clip: float = 10.0) -> PreparedData:
    """Bölmeyi yapar, ölçekleyiciyi yalnızca train'e fit eder ve tüm paneli ölçekler.

    spec: SplitSpec ya da config sözlüğü.
    """
    splits = make_splits(panel, spec, lookback)
    scaler = Standardizer(clip=clip).fit(panel, splits)
    return PreparedData(panel=panel, splits=splits, scaler=scaler,
                        X=scaler.transform_X(panel.X), y=scaler.transform_y(panel.y))


__all__ = ["DEFAULT_EMBARGO", "SPLIT_NAMES", "UNUSED", "PreparedData", "SplitSpec", "Splits",
           "Standardizer", "WindowDataset", "make_splits", "prepare", "walk_forward_specs"]

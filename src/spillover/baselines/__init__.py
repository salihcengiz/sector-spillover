"""Kıyas modelleri: persistence, EWMA, HAR-kendi, HAR+piyasa, Ridge-tüm.

Tipik kullanım:
    preds = fit_predict(panel, splits)              # {ad: (T, N) tahmin}
    preds = fit_predict(panel, splits, ["har_own"], params={"har_own": {"windows": [1, 5, 22]}})
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ..data.panel import Panel
from ..prep.splits import Splits
from .base import HAR_WINDOWS, Baseline, har_regressors
from .linear import HARMarket, HAROwn, RidgeAll
from .simple import EWMA, EWMA_LAMBDA, Persistence

BASELINES = {cls.name: cls for cls in (Persistence, EWMA, HAROwn, HARMarket, RidgeAll)}


def build_baseline(name: str, **params) -> Baseline:
    if name not in BASELINES:
        raise ValueError(f"Bilinmeyen kıyas modeli: {name!r}. Geçerli: {sorted(BASELINES)}.")
    return BASELINES[name](**params)


def fit_predict(panel: Panel, splits: Splits, names: Optional[Sequence[str]] = None,
                params: Optional[Dict[str, dict]] = None) -> Dict[str, np.ndarray]:
    """Seçilen kıyas modellerini train'de eğitir ve tüm dönemler için tahmin üretir."""
    params = params or {}
    out = {}
    for name in names or BASELINES:
        model = build_baseline(name, **params.get(name, {})).fit(panel, splits)
        out[name] = model.predict(panel)
    return out


__all__ = ["BASELINES", "EWMA", "EWMA_LAMBDA", "HAR_WINDOWS", "Baseline", "HARMarket", "HAROwn",
           "Persistence", "RidgeAll", "build_baseline", "fit_predict", "har_regressors"]

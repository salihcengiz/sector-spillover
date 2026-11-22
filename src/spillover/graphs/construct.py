"""Graf kurulumu: sektörler arası komşuluk matrisleri (her katmanda yalnızca train dönemi verisiyle).

Graf tipleri (config'te `type`):
  glasso : Zhang vd. (2025). Train dönemindeki günlük getiriler üzerinde GraphicalLassoCV (varsayılan ayarlar,
           standartlaştırma yok; yazarların kodundaki gibi). Kesinlik matrisinde sıfırdan farklı köşegen dışı eleman
           = kenar. İkili, yönsüz. scikit-learn gerektirir.
  corr   : Son vd. (2023). Train dönemlerindeki log RV'lerin Pearson korelasyonu; ağırlık max(r, 0), köşegen 0.
  none   : boş graf (grafın katkısını ölçmek için kontrol).

Bütün graflar GNN/GHAR'da kullanılmadan önce `normalize_adjacency` ile W = D^-1/2 A D^-1/2'ye çevrilir
(kendine bağlantı yok; bağlantısız düğümün satırı 0, yani komşu terimi yok).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..data.panel import Panel
from ..prep.splits import Splits

GRAPH_TYPES = ("glasso", "corr", "none")


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """W = D^-1/2 A D^-1/2, köşegen sıfırlanmış A ile. Negatif ağırlık kabul edilmez."""
    A = np.array(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Komşuluk matrisi kare olmalı.")
    if (A < 0).any():
        raise ValueError("Komşuluk matrisinde negatif ağırlık olamaz.")
    np.fill_diagonal(A, 0.0)
    degree = A.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    inv_sqrt[degree > 0] = degree[degree > 0] ** -0.5
    return inv_sqrt[:, None] * A * inv_sqrt[None, :]


def density(A: np.ndarray) -> float:
    """Köşegen dışı sıfırdan farklı elemanların oranı."""
    A = np.asarray(A)
    off = ~np.eye(len(A), dtype=bool)
    return float((A[off] != 0).mean())


def train_daily_returns(returns: pd.DataFrame, panel: Panel, splits: Splits) -> pd.DataFrame:
    """Günlük getirilerden yalnızca train dönemlerine düşen günleri seçer.

    Her gün, kendisini kapsayan panel dönemine (günlükte aynı gün, aylıkta o ay) eşlenir ve o dönem train
    satırlarından biriyse alınır. Böylece aylık bir panelde de graf yalnızca train aylarının günleriyle kurulur.
    """
    if list(returns.columns) != list(panel.sectors):
        raise ValueError("Getiri sütunları panelin sektörleriyle aynı sırada olmalı.")
    rows = np.searchsorted(panel.dates.values, returns.index.values, side="left")
    mask = np.isin(rows, splits.train_rows)
    return returns.loc[mask]


def glasso_adjacency(daily: pd.DataFrame) -> Dict:
    from sklearn.covariance import GraphicalLassoCV
    model = GraphicalLassoCV().fit(daily.to_numpy(dtype=float))
    A = (model.precision_ != 0).astype(float)
    np.fill_diagonal(A, 0.0)
    return {"A": A, "alpha": float(model.alpha_), "n_obs": len(daily)}


def correlation_adjacency(panel: Panel, rows: np.ndarray) -> Dict:
    C = np.corrcoef(panel.log_rv[rows].T)
    A = np.clip(C, 0.0, None)
    np.fill_diagonal(A, 0.0)
    return {"A": A, "n_obs": len(rows)}


def build_graph(spec: dict, panel: Panel, splits: Splits, returns: Optional[pd.DataFrame] = None) -> Dict:
    """Config'teki graf tanımından {"A", "W", "type", "density", ...} sözlüğü üretir."""
    kind = spec.get("type")
    if kind == "glasso":
        if returns is None:
            raise ValueError("glasso grafı için günlük getiriler gerekli.")
        out = glasso_adjacency(train_daily_returns(returns, panel, splits))
    elif kind == "corr":
        out = correlation_adjacency(panel, splits.train_rows)
    elif kind == "none":
        out = {"A": np.zeros((panel.N, panel.N))}
    else:
        raise ValueError(f"Bilinmeyen graf tipi: {kind!r}. Geçerli: {GRAPH_TYPES}.")
    out["type"] = kind
    out["W"] = normalize_adjacency(out["A"])
    out["density"] = density(out["A"])
    return out

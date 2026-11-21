"""Tahminleri aynı örnekler üzerinde karşılaştıran tablolar.

evaluate(...) üç tablo döndürür:
  summary    : model başına MSE, OOS R² (49 sektörün ortalaması), QLIKE, referansa göre göreli MSE
  per_sector : model × sektör başına MSE, OOS R², QLIKE
  dm         : her modelin referansa karşı DM testi; sector == "ALL" satırı, kaybın her dönemde
               49 sektör üzerinden ortalaması alınarak yapılan testtir.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..data.panel import Panel
from .dm import diebold_mariano
from .metrics import LOSSES, oos_r2

ALL_SECTORS = "ALL"


def evaluate(predictions: Dict[str, np.ndarray], panel: Panel, idx: np.ndarray, benchmark: np.ndarray,
             reference: Optional[str] = None, dm_lag: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """predictions: {ad: (T, N) tahmin}. idx: değerlendirilecek örnekler (örn. splits.test).
    benchmark: (N,) OOS R² kıyası (train ortalaması). reference: DM ve göreli MSE için model adı.
    """
    idx = np.asarray(idx)
    y = panel.y[idx]
    if not np.isfinite(y).all():
        raise ValueError("Değerlendirme örneklerinde hedef tanımsız.")
    sectors = list(panel.sectors)

    losses, summary, per_sector = {}, [], []
    for name, pred in predictions.items():
        p = np.asarray(pred)[idx]
        if p.shape != y.shape or not np.isfinite(p).all():
            raise ValueError(f"{name}: tahmin boyutu yanlış ya da değerlendirme örneklerinde NaN var.")
        losses[name] = {key: fn(y, p) for key, fn in LOSSES.items()}       # her biri (n, N)
        r2 = oos_r2(y, p, benchmark)
        summary.append({"model": name, "mse": losses[name]["mse"].mean(), "oos_r2": r2.mean(),
                        "qlike": losses[name]["qlike"].mean()})
        per_sector.append(pd.DataFrame({"model": name, "sector": sectors,
                                        "mse": losses[name]["mse"].mean(axis=0), "oos_r2": r2,
                                        "qlike": losses[name]["qlike"].mean(axis=0)}))

    summary = pd.DataFrame(summary).set_index("model")
    per_sector = pd.concat(per_sector, ignore_index=True)

    dm_rows = []
    if reference is not None:
        if reference not in losses:
            raise ValueError(f"Referans model {reference!r} tahminler arasında yok.")
        summary["rel_mse"] = summary["mse"] / summary.loc[reference, "mse"]
        for name in losses:
            if name == reference:
                continue
            for key in LOSSES:
                a, b = losses[name][key], losses[reference][key]
                columns = [(ALL_SECTORS, a.mean(axis=1), b.mean(axis=1))]
                columns += [(s, a[:, j], b[:, j]) for j, s in enumerate(sectors)]
                for sector, la, lb in columns:
                    res = diebold_mariano(la, lb, horizon=panel.horizon, lag=dm_lag)
                    dm_rows.append({"model": name, "reference": reference, "loss": key, "sector": sector,
                                    "stat": res.stat, "p_value": res.p_value, "mean_diff": res.mean_diff,
                                    "n": res.n, "lag": res.lag})
    dm = pd.DataFrame(dm_rows, columns=["model", "reference", "loss", "sector", "stat", "p_value",
                                        "mean_diff", "n", "lag"])
    return {"summary": summary, "per_sector": per_sector, "dm": dm}

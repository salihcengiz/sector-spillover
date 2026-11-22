"""Değerlendirme testleri: kayıpların, OOS R²'nin ve Diebold–Mariano testinin elle hesaplananla
tutarlı olduğu kontrol edilir.
"""
import math

import numpy as np
import pandas as pd

from spillover.data import DailyView
from spillover.evaluation import ALL_SECTORS, diebold_mariano, evaluate, newey_west_lrv, oos_r2, qlike


def test_qlike_properties():
    y = np.array([0.0, 1.0, -2.0])
    assert np.allclose(qlike(y, y), 0.0)
    assert (qlike(y, y + 0.5) > 0).all() and (qlike(y, y - 0.5) > 0).all()
    assert (qlike(y, y - 0.5) > qlike(y, y + 0.5)).all(), "riskin eksik tahmini daha ağır cezalanmalı"
    sigma2, h = math.exp(1.0), math.exp(0.3)
    assert math.isclose(qlike(np.array(1.0), np.array(0.3)), sigma2 / h - math.log(sigma2 / h) - 1)


def test_oos_r2_reference_points():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(50, 3))
    bench = np.array([0.1, -0.2, 0.0])
    assert np.allclose(oos_r2(y, y, bench), 1.0)
    assert np.allclose(oos_r2(y, np.broadcast_to(bench, y.shape), bench), 0.0)


def test_newey_west_matches_manual():
    rng = np.random.default_rng(1)
    d = rng.normal(size=200)
    c = d - d.mean()
    assert math.isclose(newey_west_lrv(d, 0), float(np.mean(c ** 2)))
    manual = np.mean(c ** 2) + 2 * sum((1 - k / 3) * np.sum(c[k:] * c[:-k]) / 200 for k in (1, 2))
    assert math.isclose(newey_west_lrv(d, 2), manual)


def test_diebold_mariano_behaviour():
    rng = np.random.default_rng(2)
    base = rng.normal(size=500) ** 2
    same = diebold_mariano(base, base)
    assert same.stat == 0.0 and same.p_value == 1.0
    worse = diebold_mariano(base + 0.5 + 0.1 * rng.normal(size=500), base, horizon=5)
    assert worse.stat > 5 and worse.p_value < 1e-6 and worse.lag == 4
    better = diebold_mariano(base, base + 0.5 + 0.1 * rng.normal(size=500))
    assert better.stat < 0, "negatif istatistik a modelinin daha iyi olduğunu göstermeli"
    d = rng.normal(size=500)
    plain = diebold_mariano(d, np.zeros(500), small_sample=False)
    assert math.isclose(plain.stat, d.mean() / math.sqrt(np.mean((d - d.mean()) ** 2) / 500))
    assert math.isclose(plain.p_value, math.erfc(abs(plain.stat) / math.sqrt(2)))


def test_evaluate_tables():
    dates = pd.bdate_range("2010-01-01", periods=200)
    rng = np.random.default_rng(3)
    returns = pd.DataFrame(rng.normal(size=(200, 3)).round(2), index=dates, columns=["A", "B", "C"])
    panel = DailyView().build(returns)
    idx = np.arange(30, 190)
    preds = {"ref": panel.log_rv, "noisy": panel.log_rv + rng.normal(scale=0.5, size=panel.log_rv.shape)}
    out = evaluate(preds, panel, idx, benchmark=panel.y[idx].mean(axis=0), reference="ref")

    summary = out["summary"]
    assert list(summary.index) == ["ref", "noisy"] and summary.loc["ref", "rel_mse"] == 1.0
    assert math.isclose(summary.loc["ref", "mse"], float(np.mean((panel.y[idx] - panel.log_rv[idx]) ** 2)))
    assert len(out["per_sector"]) == 2 * 3
    dm = out["dm"]
    assert set(dm["model"]) == {"noisy"} and len(dm) == 2 * (1 + 3)
    assert ALL_SECTORS in set(dm["sector"]) and (dm["lag"] == panel.horizon - 1).all()

    try:
        evaluate({"bad": np.full(panel.y.shape, np.nan)}, panel, idx, benchmark=np.zeros(3))
    except ValueError:
        return
    raise AssertionError("NaN tahmin hata vermeliydi")


def test_evaluate_accepts_per_period_benchmark():
    dates = pd.bdate_range("2010-01-01", periods=120)
    rng = np.random.default_rng(4)
    panel = DailyView().build(pd.DataFrame(rng.normal(size=(120, 2)).round(2), index=dates, columns=["A", "B"]))
    idx = np.arange(30, 110)
    bench = np.full(panel.y.shape, np.nan)
    bench[30:70], bench[70:110] = 0.1, -0.3
    out = evaluate({"m": panel.log_rv}, panel, idx, benchmark=bench)["per_sector"]
    expected = oos_r2(panel.y[idx], panel.log_rv[idx], bench[idx])
    assert np.allclose(out["oos_r2"], expected)
    try:
        evaluate({"m": panel.log_rv}, panel, np.arange(20, 110), benchmark=bench)
    except ValueError:
        return
    raise AssertionError("tanımsız kıyas hata vermeliydi")

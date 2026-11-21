"""Kıyas modeli testleri: regresörlerin geleceği görmediği, modellerin yalnızca train (Ridge için
train + val) verisinden öğrendiği ve bilinen doğrusal ilişkileri geri bulduğu kontrol edilir.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from spillover.baselines import (EWMA, HARMarket, HAROwn, Persistence, RidgeAll, build_baseline,
                                 fit_predict, har_regressors)
from spillover.data import DailyView, MonthlyView, load_returns
from spillover.evaluation import evaluate, train_mean
from spillover.prep import DEFAULT_EMBARGO, SplitSpec, make_splits

ROOT = Path(__file__).resolve().parents[1]
RETURNS_CSV = ROOT / "data" / "processed" / "49_sektor_gunluk_getiri_temiz.csv"
SPEC = SplitSpec.sequential("2009-06-30", "2010-06-30", embargo=5)


def make_returns(n_days=900, n_sectors=4, start="2008-01-02", seed=2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    data = rng.normal(0.0, 1.2, size=(n_days, n_sectors)).round(2)
    return pd.DataFrame(data, index=dates, columns=[f"S{i}" for i in range(n_sectors)])


def daily_setup():
    panel = DailyView().build(make_returns())
    return panel, make_splits(panel, SPEC, lookback=22)


def test_har_regressors_match_daily_features():
    panel, _ = daily_setup()
    R = har_regressors(panel)
    for k, name in enumerate(("log_r2", "log_rv5", "log_rv22")):
        assert np.allclose(R[..., k], panel.feature(name), equal_nan=True)
    assert np.allclose(Persistence().predict(panel), panel.feature("log_rv5"), equal_nan=True)


def test_monthly_har_and_persistence():
    panel = MonthlyView().build(make_returns(1500, start="2001-01-02"))
    R = har_regressors(panel)
    assert R.shape == (panel.T, panel.N, 3)
    assert np.allclose(R[..., 0], panel.log_rv)
    assert np.allclose(R[11:, :, 2], np.log(np.stack([panel.r2[t - 11:t + 1].mean(0) for t in range(11, panel.T)]) + panel.eps))
    assert np.allclose(Persistence().predict(panel), panel.log_rv)


def test_ewma_matches_manual_and_is_causal():
    panel, _ = daily_setup()
    pred = EWMA(lam=0.9).predict(panel)
    var = panel.r2[0].copy()
    for t in range(1, 50):
        var = 0.9 * var + 0.1 * panel.r2[t]
    assert np.allclose(pred[49], np.log(var + panel.eps))
    future = replace(panel, r2=np.concatenate([panel.r2[:300], panel.r2[300:] * 10]))
    assert np.allclose(EWMA(lam=0.9).predict(future)[:300], pred[:300])


def test_linear_models_recover_exact_relationship():
    """y tam olarak HAR regresörlerinin doğrusal bir fonksiyonuysa tahmin y'ye eşit olmalı."""
    panel, splits = daily_setup()
    R = har_regressors(panel)
    y = 0.5 + 0.3 * R[..., 0] + 0.2 * R[..., 1] + 0.4 * R[..., 2]
    y[-panel.horizon:] = np.nan
    exact = replace(panel, y=y)
    for model in (HAROwn(), HARMarket()):
        pred = model.fit(exact, splits).predict(exact)
        assert np.allclose(pred[splits.test], y[splits.test], atol=1e-8), model.name
    ridge = RidgeAll().fit(exact, splits)
    assert ridge.alpha_ == min(ridge.alphas)
    resid = ridge.predict(exact)[splits.test] - y[splits.test]
    assert np.mean(resid ** 2) < 1e-3 * np.var(y[splits.test])


def test_ridge_all_uses_other_sectors():
    """Hedef başka bir sektörün geçmişine bağlıysa yalnızca Ridge-tüm bunu yakalayabilir."""
    panel, splits = daily_setup()
    R = har_regressors(panel)
    y = np.roll(R[..., 2], 1, axis=1)                       # j sektörünün hedefi = (j-1) sektörünün log_rv22'si
    y[-panel.horizon:] = np.nan
    cross = replace(panel, y=y)
    preds = fit_predict(cross, splits, ["har_own", "ridge_all"])
    table = evaluate(preds, cross, splits.test, train_mean(cross, splits))["summary"]
    assert table.loc["ridge_all", "oos_r2"] > 0.95 > table.loc["har_own", "oos_r2"]


def test_models_ignore_targets_outside_fit_samples():
    panel, splits = daily_setup()
    y = panel.y.copy()
    y[np.setdiff1d(np.arange(panel.T), splits.train)] = 1e6
    corrupted = replace(panel, y=y)
    for name in ("har_own", "har_market"):
        a = build_baseline(name).fit(panel, splits).predict(panel)
        b = build_baseline(name).fit(corrupted, splits).predict(panel)
        assert np.allclose(a, b, equal_nan=True), name

    y = panel.y.copy()                                      # Ridge val'e de bakar (alpha seçimi)
    y[np.setdiff1d(np.arange(panel.T), np.concatenate([splits.train, splits.val]))] = 1e6
    a = RidgeAll().fit(panel, splits)
    b = RidgeAll().fit(replace(panel, y=y), splits)
    assert a.alpha_ == b.alpha_ and np.allclose(a.predict(panel), b.predict(panel), equal_nan=True)


def test_unknown_baseline_rejected():
    try:
        build_baseline("garch")
    except ValueError:
        return
    raise AssertionError("bilinmeyen kıyas modeli hata vermeliydi")


def test_real_data_baselines_split_b():
    """Kontrol noktası: B bölmesinde günlük HAR-kendi test R² ≈ 0.35 (ön analizle aynı)."""
    if not RETURNS_CSV.exists():
        return
    returns = load_returns(RETURNS_CSV, "1985-01-01", "2026-09-30")
    expected = {
        "daily": (DailyView(), 60, {"persistence": 0.013, "ewma": 0.283, "har_own": 0.354,
                                     "har_market": 0.367, "ridge_all": 0.338}),
        "monthly": (MonthlyView(), 12, {"persistence": 0.273, "ewma": 0.094, "har_own": 0.415,
                                         "har_market": 0.424, "ridge_all": 0.349}),
    }
    for frequency, (view, lookback, r2) in expected.items():
        panel = view.build(returns)
        splits = make_splits(panel, SplitSpec.sequential("2010-12-31", "2016-12-31",
                                                         embargo=DEFAULT_EMBARGO[frequency]), lookback)
        summary = evaluate(fit_predict(panel, splits), panel, splits.test, train_mean(panel, splits))["summary"]
        for name, value in r2.items():
            assert abs(summary.loc[name, "oos_r2"] - value) < 1e-3, (frequency, name, summary.loc[name, "oos_r2"])

"""Veri katmanı testleri: özelliklerin geleceği görmediği, hedeflerin doğru hizalandığı kontrol edilir.

pytest'e bağımlı değildir (sade assert); Colab'da `pytest -q` ile çalıştırılır.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from spillover.data import (
    DailyView,
    MonthlyView,
    build_view,
    load_returns,
    load_sector_names,
    trailing_mean,
)

ROOT = Path(__file__).resolve().parents[1]
RETURNS_CSV = ROOT / "data" / "processed" / "49_sektor_gunluk_getiri_temiz.csv"
SIC_TXT = ROOT / "data" / "processed" / "Siccodes49.txt"
EPS = 0.01


def make_returns(n_days=300, n_sectors=4, start="2001-01-02", seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    data = rng.normal(0.0, 1.2, size=(n_days, n_sectors)).round(2)
    return pd.DataFrame(data, index=dates, columns=[f"S{i}" for i in range(n_sectors)])


# ------------------------------------------------------------------- günlük

def test_trailing_mean_matches_pandas():
    x = make_returns(100, 3).to_numpy()
    for k in (1, 5, 22):
        expected = pd.DataFrame(x).rolling(k).mean().to_numpy()
        assert np.allclose(trailing_mean(x, k), expected, equal_nan=True)


def test_daily_features_are_causal():
    r = make_returns(200, 3).to_numpy()
    view = DailyView()
    base = view.compute_features(r)
    future_changed = r.copy()
    future_changed[121:] += 5.0
    changed = view.compute_features(future_changed)
    assert np.allclose(base[:121], changed[:121], equal_nan=True), "t ≤ 120 özellikleri gelecekten etkilenmemeli"
    assert not np.allclose(base[121:], changed[121:], equal_nan=True)


def test_daily_target_matches_manual():
    returns = make_returns(120, 3)
    panel = DailyView(horizon=5, eps=EPS).build(returns)
    r = returns.to_numpy()
    for t in (0, 30, 114):
        manual = np.log((r[t + 1:t + 6] ** 2).mean(axis=0) + EPS)
        assert np.allclose(panel.y[t], manual)
    assert np.isnan(panel.y[-5:]).all() and np.isfinite(panel.y[:-5]).all()
    assert np.allclose(panel.r2, r ** 2)
    assert panel.horizon == 5 and panel.frequency == "daily"


def test_daily_last_row_from_history_equals_full_computation():
    """Senaryo analizi son satırın özelliklerini kısa bir geçmişten yeniden hesaplar; sonuç aynı olmalı."""
    r = make_returns(150, 3).to_numpy()
    view = DailyView()
    full = view.compute_features(r)
    for t in (40, 149):
        last = view.compute_features(r[t - view.history + 1:t + 1])[-1]
        assert np.allclose(last, full[t])
    assert view.history == 22


def test_daily_unknown_feature_rejected():
    try:
        DailyView(features=("ret", "volume"))
    except ValueError:
        return
    raise AssertionError("bilinmeyen özellik hata vermeliydi")


# -------------------------------------------------------------------- aylık

def test_monthly_view_matches_manual():
    returns = make_returns(70, 2, start="2001-01-02")   # Ocak, Şubat, Mart, Nisan 2001 başı
    panel = MonthlyView(features=("mean", "max", "min", "std"), eps=EPS).build(returns)
    months = sorted(set(returns.index.to_period("M")))
    assert panel.T == len(months)
    for m, period in enumerate(months):
        block = returns[returns.index.to_period("M") == period]
        assert panel.dates[m] == block.index.max(), "dönem etiketi ayın son işlem günü olmalı"
        assert np.allclose(panel.feature("mean")[m], block.mean())
        assert np.allclose(panel.feature("max")[m], block.max())
        assert np.allclose(panel.feature("min")[m], block.min())
        assert np.allclose(panel.feature("std")[m], block.std(ddof=1))
        assert np.allclose(panel.r2[m], (block ** 2).mean())
        assert np.allclose(panel.log_rv[m], np.log((block ** 2).mean() + EPS))
    assert np.allclose(panel.y[:-1], panel.log_rv[1:]), "hedef bir sonraki ayın riski olmalı"
    assert np.isnan(panel.y[-1]).all() and panel.horizon == 1


def test_monthly_features_are_causal():
    returns = make_returns(130, 3)
    view = MonthlyView()
    base = view.build(returns)
    last_month = returns.index.to_period("M")[-1]
    changed_returns = returns.copy()
    changed_returns.loc[returns.index.to_period("M") == last_month] += 3.0
    changed = view.build(changed_returns)
    assert np.allclose(base.X[:-1], changed.X[:-1]), "önceki ayların özellikleri değişmemeli"
    assert not np.allclose(base.y[-2], changed.y[-2]), "bir önceki ayın hedefi değişmeli"


def test_build_view_factory():
    assert isinstance(build_view("daily"), DailyView)
    view = build_view("monthly", features=["mean", "max", "min", "std"])
    assert isinstance(view, MonthlyView) and view.features[-1] == "std"


# ---------------------------------------------------------- gerçek veri dosyası

def test_real_data_loads_and_both_views_build():
    if not RETURNS_CSV.exists():
        return
    returns = load_returns(RETURNS_CSV, "1985-01-01", "2026-09-30")
    assert returns.shape == (10517, 49) and not returns.isna().any().any()

    names = load_sector_names(SIC_TXT)
    assert list(names) == list(returns.columns), "sektör adları ve sırası veri setiyle aynı olmalı"
    assert names["Fin"] == "Trading"

    daily = DailyView().build(returns)
    assert daily.X.shape == (10517, 49, 4) and daily.first_valid_index() == 21
    monthly = MonthlyView().build(returns)
    assert monthly.X.shape == (501, 49, 3) and monthly.first_valid_index() == 0
    assert monthly.dates[0] == pd.Timestamp("1985-01-31") and monthly.dates[-1] == pd.Timestamp("2026-09-30")


def test_missing_values_rejected_unless_allowed():
    if not RETURNS_CSV.exists():
        return
    try:
        load_returns(RETURNS_CSV, "1926-07-01", "1930-12-31")
    except ValueError:
        pass
    else:
        raise AssertionError("eksik hücreli dönem hata vermeliydi")
    early = load_returns(RETURNS_CSV, "1926-07-01", "1930-12-31", allow_missing=True)
    assert early.isna().any().any()

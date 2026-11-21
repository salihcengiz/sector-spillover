"""Hazırlık katmanı testleri: bölme sınırlarında sızıntı olmadığı (purge + embargo), ölçekleyicinin
yalnızca train verisine bakıldığı ve pencerelerin doğru hizalandığı kontrol edilir.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from spillover.data import DailyView, MonthlyView, load_returns
from spillover.prep import DEFAULT_EMBARGO, SplitSpec, Standardizer, WindowDataset, make_splits, prepare

ROOT = Path(__file__).resolve().parents[1]
RETURNS_CSV = ROOT / "data" / "processed" / "49_sektor_gunluk_getiri_temiz.csv"


def make_returns(n_days=900, n_sectors=4, start="2008-01-02", seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    data = rng.normal(0.0, 1.2, size=(n_days, n_sectors)).round(2)
    return pd.DataFrame(data, index=dates, columns=[f"S{i}" for i in range(n_sectors)])


def check_split_invariants(panel, splits):
    spec = splits.spec
    labels = spec.labels(panel.dates)
    is_eval = (labels == 1) | (labels == 2)
    first, h, E = panel.first_valid_index(), panel.horizon, spec.embargo
    for code, name in enumerate(("train", "val", "test")):
        idx = splits[name]
        assert idx.size > 0
        assert (labels[idx] == code).all(), f"{name}: örnek kendi döneminde olmalı"
        for k in range(1, h + 1):
            assert (idx + k < panel.T).all() and (labels[idx + k] == code).all(), f"{name}: purge ihlali"
        assert (idx - splits.lookback + 1 >= first).all(), f"{name}: girdi penceresi tanımlı verinin içinde olmalı"
        assert np.isfinite(panel.y[idx]).all()
    for t in splits.train:
        lo, hi = max(t - E, 0), min(t + h + E, panel.T - 1)
        assert not is_eval[lo:hi + 1].any(), f"train örneği {t}: embargo ihlali"
    for t in splits.train_rows:
        assert labels[t] == 0 and not is_eval[max(t - E, 0):t + 1].any()


# ------------------------------------------------------------- SplitSpec

def test_sequential_spec_and_config_are_equivalent():
    a = SplitSpec.sequential("2010-12-31", "2016-12-31", embargo=3)
    b = SplitSpec.from_config({"train_end": "2010-12-31", "val_end": "2016-12-31", "embargo": 3})
    c = SplitSpec.from_config({"train": [[None, "2010-12-31"]], "val": [["2011-01-01", "2016-12-31"]],
                               "test": [["2017-01-01", None]], "embargo": 3, "name": "sequential"})
    assert a == b == c


def test_overlapping_ranges_rejected():
    spec = SplitSpec(train=[(None, "2009-06-30")], val=[("2009-06-01", "2009-12-31")], test=[("2010-01-01", None)])
    try:
        spec.labels(pd.bdate_range("2009-01-01", "2010-06-30"))
    except ValueError:
        return
    raise AssertionError("çakışan aralıklar hata vermeliydi")


# --------------------------------------------------------- sıralı bölme (B)

def test_daily_sequential_splits_synthetic():
    panel = DailyView().build(make_returns())
    for embargo in (0, 5):
        splits = make_splits(panel, SplitSpec.sequential("2009-06-30", "2010-06-30", embargo=embargo), lookback=20)
        check_split_invariants(panel, splits)
    no_gap = make_splits(panel, SplitSpec.sequential("2009-06-30", "2010-06-30"), lookback=20)
    assert splits.train.max() == no_gap.train.max() - 5, "embargo train sonundan tam E örnek kesmeli"
    assert np.array_equal(splits.val, no_gap.val) and np.array_equal(splits.test, no_gap.test)


def test_monthly_sequential_splits_synthetic():
    panel = MonthlyView().build(make_returns(n_days=1500, start="2001-01-02"))
    splits = make_splits(panel, SplitSpec.sequential("2003-12-31", "2004-12-31", embargo=1), lookback=6)
    check_split_invariants(panel, splits)


# ---------------------------------- araya giren pencereler (kriz / walk-forward)

def test_embargo_on_both_sides_of_a_window():
    """Train test penceresinin iki yanında da varsa, iki yanda da E + purge kadar boşluk kalmalı."""
    panel = DailyView().build(make_returns())
    E, h = 10, panel.horizon
    spec = SplitSpec(train=[(None, "2008-12-31"), ("2009-08-01", None)],
                     val=[("2009-01-01", "2009-04-30")], test=[("2009-05-01", "2009-07-31")], embargo=E)
    splits = make_splits(panel, spec, lookback=20)
    check_split_invariants(panel, splits)

    labels = spec.labels(panel.dates)
    eval_pos = np.flatnonzero(labels >= 1)
    train = splits.train
    assert train[train < eval_pos.min()].max() == eval_pos.min() - h - E - 1
    assert train[train > eval_pos.max()].min() == eval_pos.max() + E + 1
    assert splits.train_rows[splits.train_rows > eval_pos.max()].min() == eval_pos.max() + E + 1


def test_unused_periods_are_ignored():
    """Kriz katmanı: test penceresinden sonrası hiçbir bölüme girmez."""
    panel = DailyView().build(make_returns())
    spec = SplitSpec(train=[(None, "2008-12-31")], val=[("2009-01-01", "2009-06-30")],
                     test=[("2009-07-01", "2009-09-30")], embargo=DEFAULT_EMBARGO["daily"])
    splits = make_splits(panel, spec, lookback=20)
    check_split_invariants(panel, splits)
    last_test_day = panel.dates[splits.test.max()]
    assert last_test_day <= pd.Timestamp("2009-09-30") - pd.offsets.BDay(panel.horizon)
    assert panel.dates[np.concatenate([splits.train, splits.val, splits.test])].max() == last_test_day


# ------------------------------------------------------------- ölçekleme

def test_scaler_ignores_everything_outside_train():
    panel = DailyView().build(make_returns())
    splits = make_splits(panel, SplitSpec.sequential("2009-06-30", "2010-06-30", embargo=5), lookback=20)
    reference = Standardizer().fit(panel, splits)

    outside_rows = np.setdiff1d(np.arange(panel.T), splits.train_rows)
    outside_targets = np.setdiff1d(np.arange(panel.T), splits.train)
    X, y = panel.X.copy(), panel.y.copy()
    X[outside_rows[outside_rows >= panel.first_valid_index()]] = 1e6
    y[outside_targets] = 1e6
    corrupted = Standardizer().fit(replace(panel, X=X, y=y), splits)

    for key in ("x_mean", "x_std", "y_mean", "y_std"):
        assert np.allclose(getattr(reference, key), getattr(corrupted, key)), key


def test_scaler_roundtrip_and_clip():
    panel = DailyView().build(make_returns())
    data = prepare(panel, SplitSpec.sequential("2009-06-30", "2010-06-30"), lookback=20, clip=3.0)
    finite = np.isfinite(panel.y)
    assert np.allclose(data.scaler.inverse_y(data.y)[finite], panel.y[finite], atol=1e-5)
    valid_rows = data.X[panel.first_valid_index():]
    assert np.nanmax(np.abs(valid_rows)) <= 3.0 + 1e-6


def test_window_alignment_and_batches():
    X = np.arange(30 * 2 * 1, dtype=float).reshape(30, 2, 1)
    y = np.arange(30 * 2, dtype=float).reshape(30, 2)
    ds = WindowDataset(X, y, indices=np.array([4, 10, 29]), lookback=5)
    xb, yb = ds.get([1])
    assert np.array_equal(xb[0], X[6:11]) and np.array_equal(yb[0], y[10])
    seen = np.concatenate([b[1][:, 0] for b in ds.batches(2, shuffle=True, rng=np.random.default_rng(0))])
    assert sorted(seen) == sorted(y[[4, 10, 29], 0]), "her örnek bir kez görülmeli"


# ---------------------------------------------------------- gerçek veri (B)

def test_real_data_sequential_split_sizes():
    """B düzeni (train ≤ 2010, val 2011–2016, test 2017 → 2026-09) gerçek veride."""
    if not RETURNS_CSV.exists():
        return
    returns = load_returns(RETURNS_CSV, "1985-01-01", "2026-09-30")

    daily_panel = DailyView().build(returns)
    no_gap = make_splits(daily_panel, SplitSpec.sequential("2010-12-31", "2016-12-31"), lookback=60)
    assert no_gap.sizes() == {"train": 6473, "val": 1505, "test": 2444}, "embargo=0 eski bölmeyle aynı olmalı"

    daily = prepare(daily_panel, SplitSpec.sequential("2010-12-31", "2016-12-31",
                                                      embargo=DEFAULT_EMBARGO["daily"]), lookback=60)
    assert daily.splits.sizes() == {"train": 6451, "val": 1505, "test": 2444}
    check_split_invariants(daily.panel, daily.splits)
    xb, yb = daily.dataset("train").get([0, 1])
    assert xb.shape == (2, 60, 49, 4) and yb.shape == (2, 49)

    monthly = prepare(MonthlyView().build(returns),
                      {"train_end": "2010-12-31", "val_end": "2016-12-31", "embargo": DEFAULT_EMBARGO["monthly"]},
                      lookback=12)
    assert monthly.splits.sizes() == {"train": 299, "val": 71, "test": 116}
    check_split_invariants(monthly.panel, monthly.splits)
    assert monthly.panel.dates[monthly.splits.test[0]] == pd.Timestamp("2017-01-31")
    assert monthly.panel.dates[monthly.splits.test[-1]] == pd.Timestamp("2026-08-31")
    xb, yb = monthly.dataset("test").get([0])
    assert xb.shape == (1, 12, 49, 3) and yb.shape == (1, 49)

"""Graf kurulumu ve GHAR testleri: normalizasyon, grafların yalnızca train verisiyle kurulması, GHAR'ın bilinen
doğrusal ilişkiyi geri bulması ve deney akışına bağlanması. GLASSO testi scikit-learn yoksa atlanır.
"""
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from spillover.baselines import GHAR
from spillover.data import DailyView, MonthlyView
from spillover.experiment import run_experiment
from spillover.graphs import build_graph, density, normalize_adjacency, train_daily_returns
from spillover.prep import SplitSpec, make_splits

try:
    import sklearn  # noqa: F401
except ImportError:  # pragma: no cover
    sklearn = None


def make_returns(n_days=1500, n_sectors=5, start="2001-01-02", seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    common = rng.normal(0, 0.8, size=(n_days, 1))
    data = (common + rng.normal(0, 1.0, size=(n_days, n_sectors))).round(2)
    return pd.DataFrame(data, index=dates, columns=[f"S{i}" for i in range(n_sectors)])


def monthly_setup():
    returns = make_returns()
    panel = MonthlyView().build(returns)
    splits = make_splits(panel, SplitSpec.sequential("2003-12-31", "2004-12-31", embargo=1), lookback=3)
    return returns, panel, splits


# ------------------------------------------------------------- normalizasyon

def test_normalize_adjacency():
    rng = np.random.default_rng(0)
    A = rng.uniform(0, 1, size=(6, 6))
    A = (A + A.T) / 2
    A[5, :] = A[:, 5] = 0.0                                  # bağlantısız düğüm
    W = normalize_adjacency(A)
    A0 = A.copy()
    np.fill_diagonal(A0, 0)
    d = A0.sum(1)
    expected = np.zeros_like(A0)
    expected[:5, :5] = A0[:5, :5] / np.sqrt(np.outer(d[:5], d[:5]))
    assert np.allclose(W, expected) and np.allclose(W, W.T) and np.allclose(np.diag(W), 0)
    assert not W[5].any() and not W[:, 5].any(), "bağlantısız düğümün komşu terimi olmamalı"
    assert density(np.ones((4, 4))) == 1.0 and density(np.eye(4)) == 0.0
    try:
        normalize_adjacency(-A)
    except ValueError:
        return
    raise AssertionError("negatif ağırlık hata vermeliydi")


# ------------------------------------------------------- yalnızca train verisi

def test_train_daily_returns_only_train_months():
    returns, panel, splits = monthly_setup()
    daily = train_daily_returns(returns, panel, splits)
    assert daily.index.max() <= pd.Timestamp("2003-12-31")
    assert daily.index.min() == returns.index.min()
    # train, test penceresinin iki yanındaysa test ayları dışarıda kalmalı
    spec = SplitSpec(train=[(None, "2002-12-31"), ("2004-01-01", None)], val=[("2003-01-01", "2003-06-30")],
                     test=[("2003-07-01", "2003-12-31")], embargo=1)
    sp = make_splits(panel, spec, lookback=3)
    daily = train_daily_returns(returns, panel, sp)
    inside = (daily.index >= "2003-01-01") & (daily.index <= "2003-12-31")
    assert not inside.any(), "val/test aylarının günleri grafa girmemeli"
    assert daily.index.max() > pd.Timestamp("2004-01-01")


def test_correlation_graph_ignores_data_outside_train():
    _, panel, splits = monthly_setup()
    g = build_graph({"type": "corr"}, panel, splits)
    assert np.allclose(g["A"], g["A"].T) and (g["A"] >= 0).all() and np.allclose(np.diag(g["A"]), 0)
    log_rv = panel.log_rv.copy()
    outside = np.setdiff1d(np.arange(panel.T), splits.train_rows)
    log_rv[outside] = np.random.default_rng(1).normal(size=(len(outside), panel.N)) * 50
    g2 = build_graph({"type": "corr"}, replace(panel, log_rv=log_rv), splits)
    assert np.allclose(g["A"], g2["A"])
    none = build_graph({"type": "none"}, panel, splits)
    assert not none["W"].any() and none["density"] == 0.0


def test_glasso_graph():
    if sklearn is None:
        return
    returns, panel, splits = monthly_setup()
    g = build_graph({"type": "glasso"}, panel, splits, returns)
    A = g["A"]
    assert set(np.unique(A)) <= {0.0, 1.0} and np.allclose(A, A.T) and np.allclose(np.diag(A), 0)
    assert g["n_obs"] == len(train_daily_returns(returns, panel, splits)) and g["alpha"] > 0


# ------------------------------------------------------------------- GHAR

def test_ghar_recovers_exact_relationship():
    panel = DailyView().build(make_returns(n_days=600))
    splits = make_splits(panel, SplitSpec.sequential("2002-06-28", "2002-09-30"), lookback=22)
    rng = np.random.default_rng(3)
    A = (rng.uniform(size=(panel.N, panel.N)) > 0.4).astype(float)
    W = normalize_adjacency(np.maximum(A, A.T))
    X = panel.X
    beta, gamma = np.array([0.1, 0.3, -0.2, 0.5]), np.array([0.2, -0.1, 0.4, 0.05])
    alpha = rng.normal(size=panel.N)
    y = alpha + X @ beta + np.einsum("ij,tjf->tif", W, X) @ gamma
    y[-panel.horizon:] = np.nan
    exact = replace(panel, y=y)
    model = GHAR("view").fit(exact, splits, W)
    assert np.allclose(model.coef_, np.concatenate([beta, gamma]), atol=1e-8)
    assert np.allclose(model.alpha_, alpha, atol=1e-8)
    assert np.allclose(model.predict(exact)[splits.test], y[splits.test], atol=1e-8)


def test_ghar_empty_graph_is_pooled_ols_and_uses_only_train():
    _, panel, splits = monthly_setup()
    W0 = np.zeros((panel.N, panel.N))
    model = GHAR("view").fit(panel, splits, W0)
    assert model.coef_.shape == (panel.F,), "boş grafta yalnızca kendi özellikleri kullanılmalı"
    idx = splits.train
    dummies = np.kron(np.ones((len(idx), 1)), np.eye(panel.N))
    design = np.hstack([panel.X[idx].reshape(-1, panel.F), dummies])
    manual = np.linalg.lstsq(design, panel.y[idx].reshape(-1), rcond=None)[0]
    assert np.allclose(model.coef_, manual[:panel.F]) and np.allclose(model.alpha_, manual[panel.F:])

    y = panel.y.copy()
    y[np.setdiff1d(np.arange(panel.T), idx)] = 1e6
    corrupted = GHAR("view").fit(replace(panel, y=y), splits, W0)
    assert np.allclose(corrupted.coef_, model.coef_) and np.allclose(corrupted.alpha_, model.alpha_)

    har = GHAR("har").fit(panel, make_splits(panel, splits.spec, lookback=12), normalize_adjacency(np.ones((5, 5))))
    assert har.coef_.shape == (6,)


def test_experiment_runs_graph_models():
    returns = make_returns()
    cfg = {"name": "g", "output_dir": "runs", "data": {"frequency": "monthly"},
           "split": {"train_end": "2003-12-31", "val_end": "2004-12-31", "embargo": 1},
           "prep": {"lookback": 12}, "baselines": {"names": ["har_own"]},
           "graph_models": {"graphs": {"corr": {"type": "corr"}, "none": {"type": "none"}},
                            "models": [{"name": "ghar_view_corr", "type": "ghar", "features": "view", "graph": "corr"},
                                       {"name": "ghar_view_none", "type": "ghar", "features": "view", "graph": "none"},
                                       {"name": "ghar_har_corr", "type": "ghar", "features": "har", "graph": "corr"}]},
           "models": [], "evaluation": {"reference": "har_own"}}
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = run_experiment(cfg, root=Path(tmp), returns=returns, log=lambda _: None)
        summary = pd.read_csv(run_dir / "main" / "summary.csv", index_col=0)
        assert {"ghar_view_corr", "ghar_view_none", "ghar_har_corr"} <= set(summary.index)
        graphs = pd.read_csv(run_dir / "main" / "graphs.csv")
        assert list(graphs["graph"]) == ["corr", "none"] and graphs.loc[1, "density"] == 0.0
        stored = np.load(run_dir / "main" / "graphs.npz")
        assert stored["A__corr"].shape == (5, 5)

"""Deney akışı testleri: config'teki katman tanımları ve uçtan uca çıktı dosyaları.

Yalnızca kıyas modelleriyle çalışan testler torch gerektirmez. LSTM'li uçtan uca test torch yoksa atlanır.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from spillover.evaluation import evaluate
from spillover.experiment import POOLED, apply_quick, run_experiment, split_folds
from spillover.prep import SplitSpec

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "name": "test",
    "output_dir": "runs",
    "data": {"frequency": "daily", "view": {"horizon": 5}},
    "split": {"embargo": 5, "folds": [
        {"name": "a", "train": [[None, "2008-12-31"]], "val": [["2009-01-01", "2009-06-30"]],
         "test": [["2009-07-01", "2009-12-31"]]},
        {"name": "b", "train": [[None, "2009-06-30"]], "val": [["2009-07-01", "2009-12-31"]],
         "test": [["2010-01-01", "2010-06-30"]]},
    ]},
    "prep": {"lookback": 22},
    "baselines": {"names": ["persistence", "har_own", "har_market"]},
    "models": [],
    "evaluation": {"reference": "har_own"},
}


def make_returns(n_days=700, n_sectors=4, seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2008-01-02", periods=n_days)
    return pd.DataFrame(rng.normal(0, 1.2, size=(n_days, n_sectors)).round(2), index=dates,
                        columns=[f"S{i}" for i in range(n_sectors)])


def test_split_folds_from_config():
    folds = split_folds(CONFIG["split"])
    assert [n for n, _ in folds] == ["a", "b"] and all(s.embargo == 5 for _, s in folds)
    wf = split_folds({"embargo": 1, "walk_forward": {"first_test_year": 2017, "last_test_year": 2026, "val_years": 6}})
    assert [n for n, _ in wf] == [f"y{y}" for y in range(2017, 2027)] and all(s.embargo == 1 for _, s in wf)
    (name, spec), = split_folds({"train_end": "2010-12-31", "val_end": "2016-12-31", "embargo": 1})
    assert name == "main" and spec == SplitSpec.sequential("2010-12-31", "2016-12-31", embargo=1, name="main")
    try:
        split_folds({"folds": [CONFIG["split"]["folds"][0]] * 2})
    except ValueError:
        return
    raise AssertionError("aynı adlı katmanlar hata vermeliydi")


def test_apply_quick():
    cfg = apply_quick({"name": "x", "seeds": [3, 4, 5], "training": {"epochs": 100}})
    assert cfg["seeds"] == [3] and cfg["training"]["epochs"] == 2 and cfg["name"] == "x_quick"


def test_real_configs_parse():
    try:
        import yaml  # noqa: F401
    except ImportError:
        return
    from spillover.experiment import load_config
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        cfg = load_config(path)
        assert split_folds(cfg["split"]), path.name


def test_baselines_only_run_writes_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = run_experiment(CONFIG, root=Path(tmp), returns=make_returns(), log=lambda _: None)
        for fold in ("a", "b"):
            d = run_dir / fold
            for name in ("summary.csv", "per_sector.csv", "dm.csv", "predictions.npz", "scaler.npz"):
                assert (d / name).exists(), (fold, name)
            summary = pd.read_csv(d / "summary.csv", index_col=0)
            assert list(summary.index) == ["persistence", "har_own", "har_market"]
            assert np.isclose(summary.loc["har_own", "rel_mse"], 1.0)
            preds = np.load(d / "predictions.npz")
            assert preds["y"].shape == preds["pred__har_own"].shape
        all_ = pd.read_csv(run_dir / "summary_all.csv")
        assert set(all_["fold"]) == {"a", "b", POOLED}

        # Birleşik değerlendirme: iki katmanın test örnekleri, her biri kendi train ortalamasıyla.
        pooled = np.load(run_dir / POOLED / "predictions.npz")
        fold_idx = [np.load(run_dir / f / "predictions.npz")["idx"] for f in ("a", "b")]
        assert np.array_equal(pooled["idx"], np.concatenate(fold_idx))
        summary = pd.read_csv(run_dir / POOLED / "summary.csv", index_col=0)
        y, p = pooled["y"], pooled["pred__har_own"]
        assert np.isclose(summary.loc["har_own", "mse"], np.mean((y - p) ** 2))

        only_b = run_experiment(CONFIG, root=Path(tmp), returns=make_returns(), folds=["b"], log=lambda _: None)
        assert not (only_b / "a").exists() and (only_b / "b").exists()


def test_slices_and_single_fold():
    cfg = {**CONFIG, "split": CONFIG["split"]["folds"][0],
           "evaluation": {"reference": "har_own", "slices": {"q3": ["2009-07-01", "2009-09-30"],
                                                            "empty": ["2015-01-01", "2015-02-01"]}}}
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = run_experiment(cfg, root=Path(tmp), returns=make_returns(), log=lambda _: None)
        assert not (run_dir / POOLED).exists(), "tek katmanda pooled klasörü olmamalı"
        slices = pd.read_csv(run_dir / "slices.csv")
        assert set(slices["slice"]) == {"q3"}, "örneği olmayan dönem atlanmalı"
        preds = np.load(run_dir / "a" / "predictions.npz")
        dates = pd.to_datetime(preds["dates"])
        in_q3 = dates <= "2009-09-30"
        row = slices.set_index("model").loc["har_own"]
        assert row["n"] == in_q3.sum()
        assert np.isclose(row["mse"], np.mean((preds["y"][in_q3] - preds["pred__har_own"][in_q3]) ** 2))
        dm = pd.read_csv(run_dir / "slices_dm.csv")
        assert set(dm["sector"]) == {"ALL"}


def test_end_to_end_with_lstm():
    try:
        import torch  # noqa: F401
    except ImportError:
        return
    cfg = {**CONFIG, "models": [
        {"name": "lstm_cross", "type": "cross_lstm", "params": {"hidden": 4}},
        {"name": "lstm_own", "type": "own_lstm", "params": {"hidden": 4, "embed_dim": 2}}],
        "training": {"epochs": 2, "batch_size": 32, "device": "cpu"}, "seeds": [0, 1],
        "evaluation": {"reference": "har_own", "pairs": [["lstm_cross", "lstm_own"]]}}
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = run_experiment(cfg, root=Path(tmp), returns=make_returns(), folds=["a"], log=lambda _: None)
        d = run_dir / "a"
        summary = pd.read_csv(d / "summary.csv", index_col=0)
        assert {"lstm_cross", "lstm_own"} <= set(summary.index)
        seeds = pd.read_csv(d / "seeds.csv")
        assert len(seeds) == 4 and {"best_epoch", "oos_r2"} <= set(seeds.columns)
        dm = pd.read_csv(d / "dm.csv")
        assert ((dm["model"] == "lstm_cross") & (dm["reference"] == "lstm_own")).any()
        assert len(list((d / "checkpoints").glob("*.pt"))) == 4

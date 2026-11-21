"""Deney akışı: config → (her katman için) veri → kıyas modelleri → LSTM'ler → değerlendirme → çıktılar.

Config (YAML ya da sözlük) bölümleri:
  name, output_dir
  data:      returns_csv, start, end, frequency (daily | monthly), view (görünüm parametreleri)
  split:     tek bölme (train_end/val_end ya da train/val/test aralıkları + embargo)
             ya da {embargo, folds: [{name, train, val, test}, ...]}. Her katman (fold) için
             bütün akış ayrı çalışır: ölçekleyici ve modeller yalnızca o katmanın train'ini görür.
  prep:      lookback, clip
  baselines: names, params
  models:    [{name, type, params}, ...]   (boş bırakılırsa yalnızca kıyas modelleri çalışır; torch gerekmez)
  training:  TrainConfig alanları
  seeds:     [0, 1, ...]   her model her seed ile ayrı eğitilir; tahminlerin ortalaması da bir model olarak raporlanır
  evaluation: reference (DM ve göreli MSE için), pairs ([[a, b], ...] ek DM karşılaştırmaları)

Çıktılar: <output_dir>/<zaman>_<name>/<katman>/ altında
  summary.csv, per_sector.csv, dm.csv, seeds.csv, history.csv, predictions.npz, scaler.npz,
  checkpoints/<model>_s<seed>.pt; kökte config.yaml (ya da config.json) ve tüm katmanların summary_all.csv'si.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .baselines import fit_predict
from .data import build_view, load_returns
from .evaluation import evaluate, train_mean
from .prep import SplitSpec, prepare


def load_config(path) -> dict:
    import yaml
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def split_folds(split_cfg: dict) -> List[Tuple[str, SplitSpec]]:
    """Config'teki split bölümünden (katman adı, SplitSpec) listesi üretir."""
    split_cfg = dict(split_cfg)
    if "folds" not in split_cfg:
        name = split_cfg.pop("name", "main")
        return [(name, SplitSpec.from_config({**split_cfg, "name": name}))]
    embargo = split_cfg.get("embargo", 0)
    folds = []
    for i, fold in enumerate(split_cfg["folds"]):
        fold = {"embargo": embargo, **fold}
        fold.setdefault("name", f"fold{i}")
        folds.append((fold["name"], SplitSpec.from_config(fold)))
    names = [n for n, _ in folds]
    if len(set(names)) != len(names):
        raise ValueError(f"Katman adları benzersiz olmalı: {names}.")
    return folds


def apply_quick(cfg: dict) -> dict:
    """Hızlı uçtan uca deneme: tek seed, 2 epoch."""
    cfg = copy.deepcopy(cfg)
    cfg["seeds"] = list(cfg.get("seeds", [0]))[:1]
    cfg.setdefault("training", {})
    cfg["training"]["epochs"] = min(int(cfg["training"].get("epochs", 2)), 2)
    cfg["name"] = f"{cfg.get('name', 'run')}_quick"
    return cfg


def _write_config(cfg: dict, path: Path) -> None:
    try:
        import yaml
        path.with_suffix(".yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    except ImportError:
        path.with_suffix(".json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _train_models(cfg: dict, data, fold_dir: Path, log: Callable[[str], None]):
    """Her model × seed için eğitim. Dönüş: ({model: ensemble tahmini}, {model/s<seed>: tahmin}, history, seed bilgisi)."""
    import torch
    from .models import build_model
    from .training import TrainConfig, predict_panel, set_seed, train_model

    panel, splits = data.panel, data.splits
    seeds = list(cfg.get("seeds", [0]))
    ensemble, per_seed, history, info = {}, {}, [], []
    (fold_dir / "checkpoints").mkdir(exist_ok=True)
    for spec in cfg.get("models", []):
        name, params = spec["name"], dict(spec.get("params", {}))
        seed_preds = []
        for seed in seeds:
            set_seed(seed)
            model = build_model(spec["type"], panel.N, panel.F, **params)
            train_cfg = TrainConfig(**{**cfg.get("training", {}), "seed": seed})
            start = time.time()
            result = train_model(model, data, train_cfg, log=log)
            pred = predict_panel(model, data, splits.test, train_cfg.eval_batch_size)
            seed_preds.append(pred)
            per_seed[f"{name}/s{seed}"] = pred
            history += [{"model": name, "seed": seed, **row} for row in result.history]
            info.append({"model": name, "seed": seed, "best_epoch": result.best_epoch,
                         "best_val_loss": result.best_val_loss, "seconds": time.time() - start})
            torch.save({"type": spec["type"], "params": params, "n_sectors": panel.N, "n_features": panel.F,
                        "seed": seed, "state_dict": model.state_dict()},
                       fold_dir / "checkpoints" / f"{name}_s{seed}.pt")
            log(f"  {name} seed {seed}: en iyi epoch {result.best_epoch}, val {result.best_val_loss:.4f}, "
                f"{time.time() - start:.0f} s")
        ensemble[name] = np.mean(seed_preds, axis=0)
    return ensemble, per_seed, history, info


def run_fold(cfg: dict, returns: pd.DataFrame, fold_name: str, spec: SplitSpec, fold_dir: Path,
             log: Callable[[str], None] = print) -> Dict[str, pd.DataFrame]:
    fold_dir.mkdir(parents=True, exist_ok=True)
    data_cfg, prep_cfg = cfg["data"], cfg.get("prep", {})
    panel = build_view(data_cfg["frequency"], **data_cfg.get("view", {})).build(returns)
    data = prepare(panel, spec, lookback=int(prep_cfg.get("lookback", 60)), clip=prep_cfg.get("clip", 10.0))
    splits = data.splits
    log(f"[{fold_name}] {panel.frequency}, örnekler {splits.sizes()}, embargo {spec.embargo}")

    base_cfg = cfg.get("baselines", {})
    preds = fit_predict(panel, splits, base_cfg.get("names"), base_cfg.get("params"))
    per_seed, history, info = {}, [], []
    if cfg.get("models"):
        lstm_preds, per_seed, history, info = _train_models(cfg, data, fold_dir, log)
        preds.update(lstm_preds)

    bench = train_mean(panel, splits)
    eval_cfg = cfg.get("evaluation", {})
    tables = evaluate(preds, panel, splits.test, bench, reference=eval_cfg.get("reference"))
    extra_dm = [evaluate({a: preds[a], b: preds[b]}, panel, splits.test, bench, reference=b)["dm"]
                for a, b in eval_cfg.get("pairs", []) if a in preds and b in preds]
    tables["dm"] = pd.concat([tables["dm"], *extra_dm], ignore_index=True)

    seeds = pd.DataFrame(info)
    if per_seed:
        seed_summary = evaluate(per_seed, panel, splits.test, bench)["summary"].reset_index(drop=True)
        seeds = pd.concat([seeds.reset_index(drop=True), seed_summary], axis=1)
    tables["seeds"] = seeds
    tables["history"] = pd.DataFrame(history)

    tables["summary"].to_csv(fold_dir / "summary.csv")
    for key in ("per_sector", "dm", "seeds", "history"):
        tables[key].to_csv(fold_dir / f"{key}.csv", index=False)
    test = splits.test
    np.savez_compressed(fold_dir / "predictions.npz", dates=panel.dates[test].strftime("%Y-%m-%d").to_numpy(),
                        idx=test, sectors=np.array(panel.sectors), y=panel.y[test],
                        **{f"pred__{k.replace('/', '__')}": v[test] for k, v in {**preds, **per_seed}.items()})
    np.savez(fold_dir / "scaler.npz", **{k: np.asarray(v, dtype=float) for k, v in data.scaler.state_dict().items()
                                         if v is not None})
    log(f"[{fold_name}] test sonuçları:\n{tables['summary'].round(4).to_string()}")
    return tables


def run_experiment(cfg: dict, root: Path = Path("."), quick: bool = False, returns: Optional[pd.DataFrame] = None,
                   folds: Optional[List[str]] = None, log: Callable[[str], None] = print) -> Path:
    """Bütün katmanları çalıştırır ve çıktı klasörünü döndürür. returns verilirse dosya okunmaz (testler için)."""
    cfg = apply_quick(cfg) if quick else copy.deepcopy(cfg)
    root = Path(root)
    data_cfg = cfg["data"]
    if returns is None:
        returns = load_returns(root / data_cfg["returns_csv"], data_cfg.get("start"), data_cfg.get("end"))

    base = root / cfg.get("output_dir", "outputs/runs") / f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg.get('name', 'run')}"
    run_dir, k = base, 1
    while run_dir.exists():                      # aynı saniyede başlayan koşular birbirinin üstüne yazmasın
        k += 1
        run_dir = base.with_name(f"{base.name}_{k}")
    run_dir.mkdir(parents=True)
    _write_config(cfg, run_dir / "config")

    summaries = []
    for fold_name, spec in split_folds(cfg["split"]):
        if folds and fold_name not in folds:
            continue
        tables = run_fold(cfg, returns, fold_name, spec, run_dir / fold_name, log)
        summaries.append(tables["summary"].reset_index().assign(fold=fold_name))
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(run_dir / "summary_all.csv", index=False)
    log(f"Çıktılar: {run_dir}")
    return run_dir

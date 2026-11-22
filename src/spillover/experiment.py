"""Deney akışı: config → (her katman için) veri → kıyas modelleri → LSTM'ler → değerlendirme → çıktılar.

Config (YAML ya da sözlük) bölümleri:
  name, output_dir
  data:      returns_csv, start, end, frequency (daily | monthly), view (görünüm parametreleri)
  split:     üç biçimden biri; her katman (fold) için bütün akış ayrı çalışır, ölçekleyici ve
             modeller yalnızca o katmanın train'ini görür:
               tek bölme      : {train_end, val_end, embargo} ya da {train, val, test, embargo}
               katman listesi : {embargo, folds: [{name, train, val, test}, ...]}
               walk-forward   : {embargo, walk_forward: {first_test_year, last_test_year, val_years}}
  prep:      lookback, clip
  baselines: names, params
  graph_models: graf tabanlı doğrusal modeller (torch gerekmez):
             {graphs: {ad: {type: glasso | corr | none}}, models: [{name, type: ghar, features: view | har, graph: ad}]}
             Graflar her katmanda yalnızca o katmanın train dönemiyle kurulur.
  models:    [{name, type, params}, ...]   (boş bırakılırsa yalnızca kıyas modelleri çalışır; torch gerekmez)
  training:  TrainConfig alanları
  seeds:     [0, 1, ...]   her model her seed ile ayrı eğitilir; tahminlerin ortalaması model olarak raporlanır
  evaluation:
    reference: DM ve göreli MSE için model adı
    pairs:     [[a, b], ...] ek DM karşılaştırmaları
    slices:    {ad: [başlangıç, bitiş], ...} birleşik test tahminleri içinden ayrıca raporlanan dönemler

Çıktılar: <output_dir>/<zaman>_<name>/ altında
  <katman>/  summary.csv, per_sector.csv, dm.csv, seeds.csv, history.csv, predictions.npz, scaler.npz,
             graphs.csv + graphs.npz (graf modelleri varsa), checkpoints/<model>_s<seed>.pt
  pooled/    birden çok katman varsa: bütün katmanların test tahminleri tek seri olarak
             (summary.csv, per_sector.csv, dm.csv, predictions.npz)
  slices.csv, slices_dm.csv   evaluation.slices verildiyse
  summary_all.csv             bütün katmanların (ve pooled'ın) özet tablosu
  config.yaml (ya da config.json)
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .baselines import GHAR, fit_predict
from .data import build_view, load_returns
from .data.panel import Panel
from .evaluation import ALL_SECTORS, evaluate, train_mean
from .graphs import build_graph
from .prep import SplitSpec, prepare, walk_forward_specs

POOLED = "pooled"


def load_config(path) -> dict:
    import yaml
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def split_folds(split_cfg: dict) -> List[Tuple[str, SplitSpec]]:
    """Config'teki split bölümünden (katman adı, SplitSpec) listesi üretir."""
    split_cfg = dict(split_cfg)
    embargo = split_cfg.get("embargo", 0)
    if "walk_forward" in split_cfg:
        return [(s.name, s) for s in walk_forward_specs(**split_cfg["walk_forward"], embargo=embargo)]
    if "folds" not in split_cfg:
        name = split_cfg.pop("name", "main")
        return [(name, SplitSpec.from_config({**split_cfg, "name": name}))]
    folds = []
    for i, fold in enumerate(split_cfg["folds"]):
        fold = {"embargo": embargo, **fold}
        fold.setdefault("name", f"fold{i}")
        folds.append((fold["name"], SplitSpec.from_config(fold)))
    names = [n for n, _ in folds]
    if len(set(names)) != len(names) or POOLED in names:
        raise ValueError(f"Katman adları benzersiz olmalı ve {POOLED!r} olmamalı: {names}.")
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


@dataclass
class FoldResult:
    tables: Dict[str, pd.DataFrame]
    predictions: Dict[str, np.ndarray]   # {model: (T, N)}; yalnızca test satırları dolu
    benchmark: np.ndarray                # (N,) bu katmanın train ortalaması
    test_idx: np.ndarray


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


def _graph_models(cfg: dict, panel: Panel, splits, returns: Optional[pd.DataFrame], fold_dir: Path,
                  log: Callable[[str], None]) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Config'teki grafları kurar (yalnızca train dönemiyle) ve graf modellerini eğitir."""
    gcfg = cfg.get("graph_models") or {}
    graphs, rows = {}, []
    for gname, gspec in gcfg.get("graphs", {}).items():
        g = build_graph(gspec, panel, splits, returns)
        graphs[gname] = g
        rows.append({"graph": gname, **{k: v for k, v in g.items() if k not in ("A", "W")}})
        log(f"  graf {gname}: {g['type']}, yoğunluk {g['density']:.3f}")
    preds = {}
    for spec in gcfg.get("models", []):
        if spec.get("type", "ghar") != "ghar":
            raise ValueError(f"Bilinmeyen graf modeli tipi: {spec.get('type')!r}.")
        if spec["graph"] not in graphs:
            raise ValueError(f"{spec['name']}: tanımsız graf {spec['graph']!r}.")
        model = GHAR(features=spec.get("features", "view")).fit(panel, splits, graphs[spec["graph"]]["W"])
        preds[spec["name"]] = model.predict(panel)
    if graphs:
        np.savez(fold_dir / "graphs.npz", **{f"A__{k}": g["A"] for k, g in graphs.items()})
    return preds, pd.DataFrame(rows)


def _evaluate_all(preds: Dict[str, np.ndarray], panel: Panel, idx: np.ndarray, bench: np.ndarray,
                  eval_cfg: dict) -> Dict[str, pd.DataFrame]:
    """Referansa karşı tablolar + config'teki ek DM çiftleri."""
    tables = evaluate(preds, panel, idx, bench, reference=eval_cfg.get("reference"))
    extra_dm = [evaluate({a: preds[a], b: preds[b]}, panel, idx, bench, reference=b)["dm"]
                for a, b in eval_cfg.get("pairs", []) if a in preds and b in preds]
    tables["dm"] = pd.concat([tables["dm"], *extra_dm], ignore_index=True)
    return tables


def _save_predictions(path: Path, panel: Panel, idx: np.ndarray, preds: Dict[str, np.ndarray]) -> None:
    # Tarih ve sektör adları düz metin dizisi (object değil): np.load allow_pickle olmadan açabilsin.
    np.savez_compressed(path, dates=np.asarray(panel.dates[idx].strftime("%Y-%m-%d"), dtype=str), idx=idx,
                        sectors=np.asarray(panel.sectors, dtype=str), y=panel.y[idx],
                        **{f"pred__{k.replace('/', '__')}": v[idx] for k, v in preds.items()})


def _save_tables(tables: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    tables["summary"].to_csv(out_dir / "summary.csv")
    for key in ("per_sector", "dm", "seeds", "history", "graphs"):
        if key in tables:
            tables[key].to_csv(out_dir / f"{key}.csv", index=False)


def run_fold(cfg: dict, panel: Panel, fold_name: str, spec: SplitSpec, fold_dir: Path,
             log: Callable[[str], None] = print, returns: Optional[pd.DataFrame] = None) -> FoldResult:
    fold_dir.mkdir(parents=True, exist_ok=True)
    prep_cfg = cfg.get("prep", {})
    data = prepare(panel, spec, lookback=int(prep_cfg.get("lookback", 60)), clip=prep_cfg.get("clip", 10.0))
    splits = data.splits
    log(f"[{fold_name}] {panel.frequency}, örnekler {splits.sizes()}, embargo {spec.embargo}")

    base_cfg = cfg.get("baselines", {})
    preds = fit_predict(panel, splits, base_cfg.get("names"), base_cfg.get("params"))
    graph_preds, graph_table = _graph_models(cfg, panel, splits, returns, fold_dir, log)
    preds.update(graph_preds)
    per_seed, history, info = {}, [], []
    if cfg.get("models"):
        lstm_preds, per_seed, history, info = _train_models(cfg, data, fold_dir, log)
        preds.update(lstm_preds)

    bench = train_mean(panel, splits)
    tables = _evaluate_all(preds, panel, splits.test, bench, cfg.get("evaluation", {}))
    seeds = pd.DataFrame(info)
    if per_seed:
        seed_summary = evaluate(per_seed, panel, splits.test, bench)["summary"].reset_index(drop=True)
        seeds = pd.concat([seeds.reset_index(drop=True), seed_summary], axis=1)
    tables["seeds"] = seeds
    tables["history"] = pd.DataFrame(history)
    if len(graph_table):
        tables["graphs"] = graph_table

    _save_tables(tables, fold_dir)
    _save_predictions(fold_dir / "predictions.npz", panel, splits.test, {**preds, **per_seed})
    np.savez(fold_dir / "scaler.npz", **{k: np.asarray(v, dtype=float) for k, v in data.scaler.state_dict().items()
                                         if v is not None})
    log(f"[{fold_name}] test sonuçları:\n{tables['summary'].round(4).to_string()}")
    # Katmanlar arası birleştirme için yalnızca test satırları tutulur.
    test_only = {}
    for name, p in preds.items():
        q = np.full(p.shape, np.nan)
        q[splits.test] = p[splits.test]
        test_only[name] = q
    return FoldResult(tables=tables, predictions=test_only, benchmark=bench, test_idx=splits.test)


def pool_folds(results: Dict[str, FoldResult], shape: Tuple[int, int]):
    """Katmanların test tahminlerini tek (T, N) seriye birleştirir. Test dönemleri çakışamaz.

    Dönüş: (tahminler, kıyas (T, N): her satırda o katmanın train ortalaması, birleşik örnek indeksleri).
    """
    idx = np.concatenate([r.test_idx for r in results.values()])
    if len(np.unique(idx)) != len(idx):
        raise ValueError("Katmanların test dönemleri çakışıyor; birleştirilemez.")
    names = set.intersection(*(set(r.predictions) for r in results.values()))
    preds = {n: np.full(shape, np.nan) for n in names}
    bench = np.full(shape, np.nan)
    for r in results.values():
        bench[r.test_idx] = r.benchmark
        for n in names:
            preds[n][r.test_idx] = r.predictions[n][r.test_idx]
    order = list(next(iter(results.values())).predictions)          # model sırasını koru
    return {n: preds[n] for n in order if n in preds}, bench, np.sort(idx)


def evaluate_slices(preds: Dict[str, np.ndarray], panel: Panel, idx: np.ndarray, bench: np.ndarray,
                    slices: Dict[str, list], eval_cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Belirli dönemler (örn. COVID) için özet ve 49 sektör ortalamalı DM tabloları."""
    summaries, dms = [], []
    for name, (start, end) in slices.items():
        dates = panel.dates[idx]
        sub = idx[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
        if len(sub) < 2:
            continue
        tables = _evaluate_all(preds, panel, sub, bench, eval_cfg)
        summaries.append(tables["summary"].reset_index().assign(slice=name, n=len(sub)))
        dm = tables["dm"]
        dms.append(dm[dm["sector"] == ALL_SECTORS].assign(slice=name))
    return (pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(),
            pd.concat(dms, ignore_index=True) if dms else pd.DataFrame())


def run_experiment(cfg: dict, root: Path = Path("."), quick: bool = False, returns: Optional[pd.DataFrame] = None,
                   folds: Optional[List[str]] = None, log: Callable[[str], None] = print) -> Path:
    """Bütün katmanları çalıştırır ve çıktı klasörünü döndürür. returns verilirse dosya okunmaz (testler için)."""
    cfg = apply_quick(cfg) if quick else copy.deepcopy(cfg)
    root = Path(root)
    data_cfg = cfg["data"]
    if returns is None:
        returns = load_returns(root / data_cfg["returns_csv"], data_cfg.get("start"), data_cfg.get("end"))
    panel = build_view(data_cfg["frequency"], **data_cfg.get("view", {})).build(returns)

    base = root / cfg.get("output_dir", "outputs/runs") / f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg.get('name', 'run')}"
    run_dir, k = base, 1
    while run_dir.exists():                      # aynı saniyede başlayan koşular birbirinin üstüne yazmasın
        k += 1
        run_dir = base.with_name(f"{base.name}_{k}")
    run_dir.mkdir(parents=True)
    _write_config(cfg, run_dir / "config")

    results: Dict[str, FoldResult] = {}
    for fold_name, spec in split_folds(cfg["split"]):
        if folds and fold_name not in folds:
            continue
        results[fold_name] = run_fold(cfg, panel, fold_name, spec, run_dir / fold_name, log, returns=returns)
    if not results:
        raise ValueError(f"Çalıştırılacak katman yok (seçilen: {folds}).")

    summaries = [r.tables["summary"].reset_index().assign(fold=name) for name, r in results.items()]
    eval_cfg = cfg.get("evaluation", {})
    if len(results) > 1:
        preds, bench, idx = pool_folds(results, panel.y.shape)
        pooled_dir = run_dir / POOLED
        pooled_dir.mkdir()
        tables = _evaluate_all(preds, panel, idx, bench, eval_cfg)
        _save_tables(tables, pooled_dir)
        _save_predictions(pooled_dir / "predictions.npz", panel, idx, preds)
        summaries.append(tables["summary"].reset_index().assign(fold=POOLED))
        log(f"[{POOLED}] {len(results)} katmanın birleşik test sonuçları ({len(idx)} örnek):\n"
            f"{tables['summary'].round(4).to_string()}")
    else:
        (only,) = results.values()
        preds, bench, idx = only.predictions, only.benchmark, only.test_idx

    if eval_cfg.get("slices"):
        slice_summary, slice_dm = evaluate_slices(preds, panel, idx, bench, eval_cfg["slices"], eval_cfg)
        slice_summary.to_csv(run_dir / "slices.csv", index=False)
        slice_dm.to_csv(run_dir / "slices_dm.csv", index=False)

    pd.concat(summaries, ignore_index=True).to_csv(run_dir / "summary_all.csv", index=False)
    log(f"Çıktılar: {run_dir}")
    return run_dir

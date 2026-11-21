"""Ortak eğitim döngüsü ve tahmin (tüm modeller ve iki frekans için aynı).

- Kayıp: ölçeklenmiş hedefte MSE (her sektör eşit ağırlıklı).
- AdamW, gradyan kırpma, val MSE'ye göre erken durdurma. Eğitim sonunda en iyi val
  epoch'unun ağırlıkları geri yüklenir.
- Örnekler PreparedData'nın train / val indekslerinden gelir; ölçekleyici yalnızca train'e fit
  edilmiştir. Mini-batch sırası seed'e bağlı bir numpy üreteciyle karıştırılır.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

from ..prep import PreparedData, WindowDataset
from .utils import resolve_device


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: Optional[float] = 1.0
    patience: int = 10
    device: str = "auto"
    seed: int = 0
    eval_batch_size: int = 1024
    log_every: int = 0          # 0: epoch logu yok; k: her k epoch'ta bir satır

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainResult:
    best_epoch: int
    best_val_loss: float
    history: List[dict] = field(default_factory=list)   # her epoch: epoch, train_loss, val_loss


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def dataset_loss(model: nn.Module, ds: WindowDataset, batch_size: int = 1024) -> float:
    """Veri kümesindeki ortalama MSE (ölçeklenmiş hedef)."""
    model.eval()
    device = _model_device(model)
    total, count = 0.0, 0
    for xb, yb in ds.batches(batch_size):
        pred = model(torch.from_numpy(xb).to(device))
        total += float(((pred - torch.from_numpy(yb).to(device)) ** 2).sum())
        count += yb.size
    return total / count


def train_model(model: nn.Module, data: PreparedData, cfg: TrainConfig,
                log: Optional[Callable[[str], None]] = None) -> TrainResult:
    device = resolve_device(cfg.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(cfg.seed)
    train_ds, val_ds = data.dataset("train"), data.dataset("val")

    best_loss, best_epoch, best_state, wait = math.inf, 0, None, 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for xb, yb in train_ds.batches(cfg.batch_size, shuffle=True, rng=rng):
            x, y = torch.from_numpy(xb).to(device), torch.from_numpy(yb).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total += float(loss) * len(xb)
            count += len(xb)
        val_loss = dataset_loss(model, val_ds, cfg.eval_batch_size)
        history.append({"epoch": epoch, "train_loss": total / count, "val_loss": val_loss})
        if log and cfg.log_every and (epoch % cfg.log_every == 0 or epoch == 1):
            log(f"  epoch {epoch:3d}  train {total / count:.4f}  val {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    if best_state is None:
        raise RuntimeError("Val kaybı hiçbir epoch'ta sonlu olmadı (epochs ≥ 1 olmalı; veriyi kontrol edin).")
    model.load_state_dict(best_state)
    return TrainResult(best_epoch=best_epoch, best_val_loss=best_loss, history=history)


@torch.no_grad()
def predict_windows(model: nn.Module, windows: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """(B, L, N, F) ölçeklenmiş pencereler → (B, N) ölçeklenmiş tahmin. Senaryo analizi de bunu kullanır."""
    model.eval()
    device = _model_device(model)
    windows = np.ascontiguousarray(windows, dtype=np.float32)
    out = [model(torch.from_numpy(windows[i:i + batch_size]).to(device)).cpu().numpy()
           for i in range(0, len(windows), batch_size)]
    return np.concatenate(out) if out else np.empty((0, windows.shape[2]), dtype=np.float32)


def predict_panel(model: nn.Module, data: PreparedData, idx: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """idx örnekleri için tahmin → (T, N), log-RV ölçeğinde (inverse_y uygulanmış); idx dışı NaN.

    Çıktı, kıyas modellerinin tahminleriyle aynı biçimdedir ve doğrudan evaluate()'e verilebilir.
    """
    idx = np.asarray(idx, dtype=np.int64)
    L = data.splits.lookback
    if idx.size and idx.min() - L + 1 < 0:
        raise ValueError("Bazı örneklerin girdi penceresi verinin başından önceye taşıyor.")
    out = np.full(data.y.shape, np.nan)
    for start in range(0, len(idx), batch_size):
        t = idx[start:start + batch_size]
        windows = data.X[t[:, None] - L + 1 + np.arange(L)]
        out[t] = data.scaler.inverse_y(predict_windows(model, windows, batch_size))
    return out

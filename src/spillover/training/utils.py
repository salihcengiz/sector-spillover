"""Cihaz seçimi ve tekrarlanabilirlik."""
from __future__ import annotations

import random

import numpy as np
import torch


def resolve_device(name: str = "auto") -> torch.device:
    """"auto": sırasıyla cuda → mps → cpu. Diğer değerler doğrudan kullanılır ("cpu", "cuda:0"…)."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Python, numpy ve torch tohumlarını ayarlar. GPU'da sonuçlar yine de bitwise aynı olmayabilir."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

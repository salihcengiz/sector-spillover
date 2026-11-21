"""Model kaydı: config'teki `type` adından model sınıfına.

Yeni bir model eklemek için models/ altına bir dosya yazılır, sınıf @register_model("ad") ile
işaretlenir ve dosya models/__init__.py içinde içe aktarılır.

Her model aynı sözleşmeye uyar:
    __init__(n_sectors, n_features, **params)
    forward(x: (B, L, N, F)) -> (B, N)     # ölçeklenmiş hedef tahmini
Model frekansı bilmez; günlük ve aylık Panel'den gelen pencereleri aynı şekilde işler.
"""
from __future__ import annotations

from typing import Callable, Dict

MODELS: Dict[str, Callable] = {}


def register_model(name: str):
    def decorator(cls):
        if name in MODELS:
            raise ValueError(f"{name!r} adında bir model zaten kayıtlı.")
        MODELS[name] = cls
        cls.model_type = name
        return cls
    return decorator


def build_model(type: str, n_sectors: int, n_features: int, **params):
    if type not in MODELS:
        raise ValueError(f"Bilinmeyen model tipi: {type!r}. Geçerli: {sorted(MODELS)}.")
    return MODELS[type](n_sectors=n_sectors, n_features=n_features, **params)

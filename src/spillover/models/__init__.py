"""Modeller. Hepsi (B, L, N, F) → (B, N) sözleşmesine uyar; seçim config'teki `type` ile yapılır.

Kayıtlı tipler: cross_lstm, own_lstm. İleride: cnn.py, gnn/ (aynı kayıt mekanizmasıyla).
"""
from .registry import MODELS, build_model, register_model
from .lstm import CrossSectorLSTM, OwnSectorLSTM

__all__ = ["MODELS", "CrossSectorLSTM", "OwnSectorLSTM", "build_model", "register_model"]

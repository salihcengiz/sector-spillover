"""Eğitim katmanı: ortak eğitim döngüsü, erken durdurma, tahmin, cihaz ve seed yönetimi."""
from .trainer import TrainConfig, TrainResult, dataset_loss, predict_panel, predict_windows, train_model
from .utils import resolve_device, set_seed

__all__ = ["TrainConfig", "TrainResult", "dataset_loss", "predict_panel", "predict_windows",
           "resolve_device", "set_seed", "train_model"]

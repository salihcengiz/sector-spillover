"""Veri katmanı: kaynak (günlük getiriler) → görünüm (günlük / aylık) → ortak Panel."""
from .panel import Panel
from .source import N_SECTORS, load_returns, load_sector_names
from .views import (
    DEFAULT_EPS,
    VIEWS,
    DailyView,
    MonthlyView,
    View,
    build_view,
    daily_features,
    trailing_mean,
)

__all__ = [
    "DEFAULT_EPS",
    "N_SECTORS",
    "VIEWS",
    "DailyView",
    "MonthlyView",
    "Panel",
    "View",
    "build_view",
    "daily_features",
    "load_returns",
    "load_sector_names",
    "trailing_mean",
]

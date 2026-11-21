"""Deney betiği: config → kıyas modelleri + LSTM'ler + değerlendirme → outputs/runs/<zaman>_<ad>/.

Colab'da (depo kökünde):
    python scripts/run_experiment.py --config configs/lstm_monthly_b.yaml --quick   # birkaç dakikalık uçtan uca deneme
    python scripts/run_experiment.py --config configs/lstm_monthly_b.yaml
    python scripts/run_experiment.py --config configs/lstm_daily_crisis.yaml --folds covid svb
    python scripts/run_experiment.py --config configs/lstm_daily_b.yaml --baselines-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    import spillover  # noqa: F401  (pip install -e . yapıldıysa)
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))

from spillover.experiment import load_config, run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="configs/ altındaki YAML dosyası")
    parser.add_argument("--quick", action="store_true", help="tek seed, 2 epoch")
    parser.add_argument("--folds", nargs="*", help="yalnızca bu katmanları çalıştır")
    parser.add_argument("--baselines-only", action="store_true", help="LSTM'leri atla (torch gerekmez)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.baselines_only:
        cfg["models"] = []
        cfg["name"] = f"{cfg.get('name', 'run')}_baselines"
    run_experiment(cfg, root=ROOT, quick=args.quick, folds=args.folds)


if __name__ == "__main__":
    main()

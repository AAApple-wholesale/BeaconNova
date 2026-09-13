from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beaconnova.strategy import run_strategy_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BeaconNova intervention strategy simulation.")
    parser.add_argument("--prediction-path", type=Path, default=Path("outputs") / "graph_v0_5_weather" / "graph_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "strategy_v0_6")
    parser.add_argument("--horizon", type=int, default=12, help="Prediction horizon in 5-minute windows; 6=30min, 12=60min.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_strategy_simulation(args.prediction_path, args.output_dir, horizon=args.horizon)
    print("Strategy simulation completed.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

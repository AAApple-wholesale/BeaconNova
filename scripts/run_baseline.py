from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beaconnova.config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, ModelConfig
from beaconnova.pipeline import run_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BeaconNova baseline model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing competition data files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output CSV files.")
    parser.add_argument("--test-days", type=int, default=21, help="Number of latest dates used as chronological test set.")
    parser.add_argument("--skip-ticket", action="store_true", help="Skip large ticket-stream aggregation for a faster smoke run.")
    parser.add_argument("--skip-facility", action="store_true", help="Skip scenic-capacity and ropeway parameter features.")
    parser.add_argument("--skip-weather", action="store_true", help="Skip weather context features.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ModelConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_days=args.test_days,
        use_ticket_features=not args.skip_ticket,
        use_facility_features=not args.skip_facility,
    )
    outputs = run_baseline(config)
    print("Baseline run completed.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()



from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beaconnova.config import DEFAULT_DATA_DIR, ModelConfig
from beaconnova.temporal_graph_model import TemporalGraphTrainConfig
from beaconnova.temporal_graph_pipeline import TemporalGraphPipelineConfig, run_temporal_graph_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BeaconNova temporal graph neural model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "temporal_graph_v0_6")
    parser.add_argument("--test-days", type=int, default=21)
    parser.add_argument("--validation-days", type=int, default=7)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--adaptive-adj-rank", type=int, default=8)
    parser.add_argument("--adaptive-adj-weight", type=float, default=0.10)
    parser.add_argument("--skip-weather", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = ModelConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_days=args.test_days,
        use_ticket_features=True,
        use_facility_features=True,
        use_weather_features=not args.skip_weather,
    )
    train_config = TemporalGraphTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        sequence_length=args.sequence_length,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        adaptive_adj_rank=args.adaptive_adj_rank,
        adaptive_adj_weight=args.adaptive_adj_weight,
    )
    outputs = run_temporal_graph_pipeline(TemporalGraphPipelineConfig(model=model_config, train=train_config, validation_days=args.validation_days))
    print("Temporal graph model run completed.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()


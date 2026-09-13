from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beaconnova.config import DEFAULT_DATA_DIR, ModelConfig
from beaconnova.ensemble import FusionTrainConfig
from beaconnova.ensemble_pipeline import EnsemblePipelineConfig, run_ensemble_pipeline
from beaconnova.graph_model import GraphTrainConfig
from beaconnova.temporal_graph_model import TemporalGraphTrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BeaconNova validation-fitted ensemble forecaster.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "ensemble_v0_9")
    parser.add_argument("--test-days", type=int, default=21)
    parser.add_argument("--validation-days", type=int, default=10)
    parser.add_argument("--graph-epochs", type=int, default=18)
    parser.add_argument("--temporal-epochs", type=int, default=20)
    parser.add_argument("--graph-hidden-dim", type=int, default=128)
    parser.add_argument("--temporal-hidden-dim", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--fusion-epochs", type=int, default=180)
    parser.add_argument("--fusion-hidden-dim", type=int, default=48)
    parser.add_argument("--fusion-patience", type=int, default=18)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-weather", action="store_true")
    parser.add_argument("--include-tabular", action="store_true", help="Also train the slower HistGradientBoosting source model.")
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
    graph_config = GraphTrainConfig(
        epochs=args.graph_epochs,
        batch_size=max(512, args.batch_size),
        hidden_dim=args.graph_hidden_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        adaptive_adj_rank=8,
        adaptive_adj_weight=0.10,
    )
    temporal_config = TemporalGraphTrainConfig(
        epochs=args.temporal_epochs,
        batch_size=args.batch_size,
        hidden_dim=args.temporal_hidden_dim,
        sequence_length=args.sequence_length,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        adaptive_adj_rank=8,
        adaptive_adj_weight=0.08,
    )
    fusion_config = FusionTrainConfig(
        epochs=args.fusion_epochs,
        batch_size=2048,
        hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        learning_rate=1e-3,
        weight_decay=args.weight_decay,
        patience=args.fusion_patience,
        device=args.device,
    )
    outputs = run_ensemble_pipeline(
        EnsemblePipelineConfig(
            model=model_config,
            graph_train=graph_config,
            temporal_train=temporal_config,
            validation_days=args.validation_days,
            fusion_train=fusion_config,
            include_tabular=args.include_tabular,
        )
    )
    print("Ensemble model run completed.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beaconnova.config import DEFAULT_DATA_DIR, ModelConfig
from beaconnova.graph_model import GraphTrainConfig
from beaconnova.graph_pipeline import GraphPipelineConfig, run_graph_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BeaconNova scenic graph neural baseline.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing competition data files.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "graph_v0_4", help="Directory for graph-model outputs.")
    parser.add_argument("--test-days", type=int, default=21, help="Number of latest dates used as chronological test set.")
    parser.add_argument("--validation-days", type=int, default=7, help="Number of latest train dates used for validation.")
    parser.add_argument("--epochs", type=int, default=16, help="Maximum graph-model training epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size.")
    parser.add_argument("--hidden-dim", type=int, default=96, help="GCN hidden dimension.")
    parser.add_argument("--dropout", type=float, default=0.12, help="Dropout ratio.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience.")
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--adaptive-adj-rank", type=int, default=8, help="Rank of learnable adaptive adjacency embeddings.")
    parser.add_argument("--adaptive-adj-weight", type=float, default=0.12, help="Blend weight for learnable adaptive adjacency.")
    parser.add_argument("--skip-weather", action="store_true", help="Skip weather context features.")
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
    train_config = GraphTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        adaptive_adj_rank=args.adaptive_adj_rank,
        adaptive_adj_weight=args.adaptive_adj_weight,
    )
    outputs = run_graph_pipeline(GraphPipelineConfig(model=model_config, train=train_config, validation_days=args.validation_days))
    print("Graph model run completed.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()




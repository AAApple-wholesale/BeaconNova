from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beaconnova.baseline import regression_metrics
from beaconnova.config import DEFAULT_DATA_DIR
from beaconnova.data import load_security_data, load_weather_context
from beaconnova.features import add_congestion_propagation_features, add_gate_context_features, add_security_history_features, add_targets, add_time_features, add_weather_features
from beaconnova.scoring import add_comfort_and_risk, risk_summary
from beaconnova.wait_specialist import fit_wait_specialist_source


TARGETS = ["target_person_h6", "target_wait_h6", "target_person_h12", "target_wait_h12"]
PREDS = ["pred_person_h6", "pred_wait_h6", "pred_person_h12", "pred_wait_h12"]


class CachedFusionGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, source_count: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.weight_head = nn.Linear(hidden_dim, source_count)
        self.bias_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, sources: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(x)
        weights = torch.softmax(self.weight_head(h), dim=1)
        bias = self.bias_head(h).squeeze(1)
        pred = torch.sum(weights * sources, dim=1) + bias
        return pred, weights, bias


def _load_sources(graph_path: Path, temporal_path: Path, data_dir: Path, horizons: tuple[int, ...]) -> pd.DataFrame:
    graph = pd.read_csv(graph_path, encoding="utf-8-sig")
    temporal = pd.read_csv(temporal_path, encoding="utf-8-sig")
    for df in (graph, temporal):
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["node_id"] = df["node_id"].astype(str)
    keys = ["datetime", "security_gate", "channel", "node_id"]
    graph_keep = keys + PREDS + [
        "total_person_count",
        "avg_queue_wait_min",
        "total_bag_check_num",
        "facility_hourly_pressure",
        "facility_bottleneck_pressure",
        "ticket_facility_hourly_pressure",
        "rail_ropeway_hourly_pressure",
        "rail_ropeway_platform_pressure",
        "ticket_ropeway_pressure",
        "weather_comfort_penalty",
        "weather_heat_stress",
        "weather_cold_stress",
        "weather_rain_flag",
        "weather_wind_stress",
    ]
    graph_keep = [col for col in graph_keep if col in graph]
    temporal_keep = keys + PREDS
    merged = graph[graph_keep].merge(temporal[temporal_keep], on=keys, how="inner", suffixes=("_gcn", "_tcn_gcn"))

    security = load_security_data(data_dir)
    security = add_time_features(security)
    security = add_gate_context_features(security)
    security = add_security_history_features(security)
    security = add_congestion_propagation_features(security)
    weather = load_weather_context(data_dir, security["datetime"])
    security = add_weather_features(security, weather)
    security = add_targets(security, horizons)
    security["node_id"] = security["node_id"].astype(str)
    extra_cols = [
        col
        for col in security.columns
        if col not in keys + ["date", "time_window"] and (col.startswith("target_") or col not in merged.columns)
    ]
    merged = merged.merge(security[keys + extra_cols], on=keys, how="inner")
    merged = merged.dropna(subset=TARGETS).sort_values(["datetime", "node_id"]).reset_index(drop=True)
    merged["hour"] = merged["datetime"].dt.hour
    merged["dayofweek"] = merged["datetime"].dt.dayofweek
    merged["slot_5min"] = merged["hour"] * 12 + (merged["datetime"].dt.minute // 5)
    merged["sin_slot"] = np.sin(2 * np.pi * merged["slot_5min"] / 288)
    merged["cos_slot"] = np.cos(2 * np.pi * merged["slot_5min"] / 288)
    merged["is_weekend"] = (merged["dayofweek"] >= 5).astype(float)
    return merged


def _feature_columns(frame: pd.DataFrame, target_idx: int) -> list[str]:
    suffix = ["_gcn", "_tcn_gcn"]
    cols = [PREDS[target_idx] + s for s in suffix]
    base = [
        "total_person_count",
        "avg_queue_wait_min",
        "total_bag_check_num",
        "total_person_count_lag_1",
        "total_person_count_lag_6",
        "total_person_count_roll_mean_6",
        "total_person_count_roll_max_12",
        "total_person_count_roll_q90_12",
        "total_person_count_peak_ratio_12",
        "person_diff_1",
        "person_momentum_30min",
        "avg_queue_wait_min_lag_1",
        "avg_queue_wait_min_lag_6",
        "avg_queue_wait_min_roll_mean_6",
        "avg_queue_wait_min_roll_max_12",
        "avg_queue_wait_min_roll_q90_12",
        "avg_queue_wait_min_peak_ratio_12",
        "wait_diff_1",
        "wait_momentum_30min",
        "wait_accel_15min",
        "gate_person_count",
        "gate_wait_mean",
        "gate_wait_max",
        "system_person_count",
        "system_wait_mean",
        "system_wait_max",
        "channel_load_share",
        "gate_system_share",
        "channel_wait_gap",
        "gate_wait_gap",
        "upstream_gate_person_lag_1",
        "upstream_gate_person_lag_6",
        "upstream_gate_wait_lag_1",
        "upstream_gate_wait_lag_6",
        "upstream_system_person_lag_6",
        "propagated_person_pressure_lag_6",
        "propagated_wait_pressure_lag_6",
        "relative_wait_pressure_lag_6",
        "facility_hourly_pressure",
        "facility_bottleneck_pressure",
        "ticket_facility_hourly_pressure",
        "rail_ropeway_hourly_pressure",
        "rail_ropeway_platform_pressure",
        "ticket_ropeway_pressure",
        "weather_comfort_penalty",
        "weather_heat_stress",
        "weather_cold_stress",
        "weather_rain_flag",
        "weather_wind_stress",
        "weather_rain_1h",
        "weather_rain_3h",
        "weather_temp_diff_1h",
        "weather_wind_diff_1h",
        "weather_penalty_diff_1h",
        "weather_rain_start_1h",
        "weather_rain_intensify_1h",
        "weather_abrupt_change_score",
        "hour",
        "dayofweek",
        "slot_5min",
        "sin_slot",
        "cos_slot",
        "sin_week",
        "cos_week",
        "is_weekend",
        "is_saturday",
        "is_sunday",
        "is_pre_holiday_3d",
        "is_post_holiday_3d",
        "is_holiday_adjacent",
        "calendar_demand_weight",
        "is_opening_ramp",
        "is_morning_peak",
        "is_midday_peak",
        "is_afternoon_peak",
        "is_closing_period",
    ]
    return [col for col in cols + base if col in frame]


def _train_one(train: pd.DataFrame, holdout: pd.DataFrame, target_col: str, pred_col: str, target_idx: int, args: argparse.Namespace, device: torch.device):
    source_cols = [pred_col + "_gcn", pred_col + "_tcn_gcn"]
    if target_col.startswith("target_wait") and pred_col + "_wait_specialist" in train.columns:
        source_cols.append(pred_col + "_wait_specialist")
    source_names = [col.rsplit("_", 1)[-1] if not col.endswith("_tcn_gcn") else "tcn_gcn" for col in source_cols]
    source_names = ["gcn" if col.endswith("_gcn") and not col.endswith("_tcn_gcn") else name for col, name in zip(source_cols, source_names)]
    source_names = ["wait_specialist" if col.endswith("_wait_specialist") else name for col, name in zip(source_cols, source_names)]
    feature_cols = _feature_columns(train, target_idx)
    train_x = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    holdout_x = holdout[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    train_x = np.ascontiguousarray(((train_x - mean) / std).astype(np.float32))
    holdout_x = np.ascontiguousarray(((holdout_x - mean) / std).astype(np.float32))
    train_sources = np.ascontiguousarray(train[source_cols].to_numpy(dtype=np.float32).copy())
    holdout_sources = np.ascontiguousarray(holdout[source_cols].to_numpy(dtype=np.float32).copy())
    train_y = np.ascontiguousarray(train[target_col].to_numpy(dtype=np.float32).copy())
    holdout_y = np.ascontiguousarray(holdout[target_col].to_numpy(dtype=np.float32).copy())

    split = max(1, int(len(train_y) * 0.82))
    train_ds = TensorDataset(torch.from_numpy(train_x[:split]), torch.from_numpy(train_sources[:split]), torch.from_numpy(train_y[:split]))
    val_x = torch.from_numpy(train_x[split:]).to(device)
    val_sources = torch.from_numpy(train_sources[split:]).to(device)
    val_y = torch.from_numpy(train_y[split:]).to(device)
    model = CachedFusionGate(train_x.shape[1], args.hidden_dim, len(source_cols), args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    best_state = None
    best_val = float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for bx, bs, by in loader:
            bx, bs, by = bx.to(device), bs.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred, _, bias = model(bx, bs)
            loss = loss_fn(pred, by) + args.correction_weight * torch.mean(bias.square())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_pred, _, val_bias = model(val_x, val_sources)
            val_loss = loss_fn(val_pred, val_y) + args.correction_weight * torch.mean(val_bias.square())
            val_mae = torch.mean(torch.abs(torch.clamp(val_pred, min=0.0) - val_y))
        history.append({"target": target_col, "epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": float(val_loss.cpu()), "val_mae": float(val_mae.cpu())})
        current = float(val_loss.cpu())
        if current < best_val - 1e-5:
            best_val = current
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred, val_weight, _ = model(val_x, val_sources)
        val_pred_np = np.clip(val_pred.cpu().numpy(), a_min=0.0, a_max=None)
    val_candidates = [("dynamic_fusion", val_pred_np, None)]
    for source_idx, source_name in enumerate(source_names):
        val_candidates.append((source_name, train_sources[split:, source_idx], source_idx))
    best_name, _, best_source_idx = min(
        val_candidates,
        key=lambda item: float(np.mean(np.abs(item[1] - train_y[split:]))),
    )

    preds = []
    weights = []
    with torch.no_grad():
        for start in range(0, len(holdout_x), args.batch_size * 4):
            bx = torch.from_numpy(holdout_x[start : start + args.batch_size * 4]).to(device)
            bs = torch.from_numpy(holdout_sources[start : start + args.batch_size * 4]).to(device)
            pred, weight, _ = model(bx, bs)
            preds.append(pred.cpu().numpy())
            weights.append(weight.cpu().numpy())
    pred = np.clip(np.concatenate(preds), a_min=0.0, a_max=None)
    weight = np.concatenate(weights)
    if args.prefer_wait_specialist and target_col.startswith("target_wait") and "wait_specialist" in source_names:
        best_source_idx = source_names.index("wait_specialist")
        best_name = "wait_specialist"
    if best_source_idx is not None:
        pred = np.clip(holdout_sources[:, best_source_idx], a_min=0.0, a_max=None)
        weight = np.zeros((len(holdout_sources), len(source_names)), dtype=np.float32)
        weight[:, best_source_idx] = 1.0
    metrics = []
    for source_name, source_col in zip(source_names, source_cols):
        item = regression_metrics(pd.Series(holdout_y), pd.Series(holdout[source_col].to_numpy(dtype=float)))
        item.update({"source": source_name, "target": target_col, "selected_by_inner_validation": best_name})
        metrics.append(item)
    item = regression_metrics(pd.Series(holdout_y), pd.Series(pred))
    item.update({"source": "dynamic_fusion", "target": target_col, "selected_by_inner_validation": best_name})
    metrics.append(item)
    weights_frame = pd.DataFrame(
        {
            "target": target_col,
            "source": source_names,
            "mean_weight": weight.mean(axis=0),
            "min_weight": weight.min(axis=0),
            "max_weight": weight.max(axis=0),
        }
    )
    return pred, pd.DataFrame(metrics), weights_frame, pd.DataFrame(history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fast dynamic fusion gate on cached GCN and TCN-GCN predictions.")
    parser.add_argument("--graph-path", type=Path, default=Path("outputs") / "graph_v0_6_optimized" / "graph_predictions.csv")
    parser.add_argument("--temporal-path", type=Path, default=Path("outputs") / "temporal_graph_v0_7_optimized" / "temporal_graph_predictions.csv")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "cached_fusion_v0_9")
    parser.add_argument("--meta-train-days", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.06)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--correction-weight", type=float, default=0.001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-prefer-wait-specialist", dest="prefer_wait_specialist", action="store_false")
    parser.set_defaults(prefer_wait_specialist=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    torch.manual_seed(42)
    np.random.seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_sources(args.graph_path, args.temporal_path, args.data_dir, (6, 12))
    dates = pd.Series(frame["datetime"].dt.normalize().unique()).sort_values()
    train_dates = set(dates.iloc[: args.meta_train_days])
    train = frame[frame["datetime"].dt.normalize().isin(train_dates)].copy()
    holdout = frame[~frame["datetime"].dt.normalize().isin(train_dates)].copy().reset_index(drop=True)
    if train.empty or holdout.empty:
        raise RuntimeError("Cached fusion split is empty; adjust --meta-train-days.")

    pred_out = holdout[["datetime", "security_gate", "channel", "node_id"]].copy()
    observed = holdout[[
        "datetime",
        "security_gate",
        "channel",
        "node_id",
        "total_person_count",
        "avg_queue_wait_min",
        "total_bag_check_num",
        *TARGETS,
    ]].copy()
    all_metrics = []
    all_weights = []
    all_history = []
    wait_importance_frames = []
    for target_col, pred_col in (("target_wait_h6", "pred_wait_h6"), ("target_wait_h12", "pred_wait_h12")):
        specialist = fit_wait_specialist_source(train, holdout, target_col, pred_col)
        train[pred_col + "_wait_specialist"] = specialist.train_prediction
        holdout[pred_col + "_wait_specialist"] = specialist.holdout_prediction
        wait_importance_frames.append(specialist.feature_importance.head(40))
    for target_idx, (target_col, pred_col) in enumerate(zip(TARGETS, PREDS)):
        pred, metrics, weights, history = _train_one(train, holdout, target_col, pred_col, target_idx, args, device)
        pred_out[pred_col] = pred
        all_metrics.append(metrics)
        all_weights.append(weights)
        all_history.append(history)

    scored = add_comfort_and_risk(pred_out, observed, (6, 12))
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics["horizon_min"] = metrics["target"].str.extract(r"_h(\d+)").astype(int) * 5
    metrics["rows"] = len(holdout)
    weights = pd.concat(all_weights, ignore_index=True)
    history = pd.concat(all_history, ignore_index=True)
    wait_importance = pd.concat(wait_importance_frames, ignore_index=True) if wait_importance_frames else pd.DataFrame()
    summary = risk_summary(scored, (6, 12))
    scored.to_csv(args.output_dir / "cached_fusion_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(args.output_dir / "cached_fusion_metrics.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(args.output_dir / "cached_fusion_weights.csv", index=False, encoding="utf-8-sig")
    history.to_csv(args.output_dir / "cached_fusion_training_history.csv", index=False, encoding="utf-8-sig")
    wait_importance.to_csv(args.output_dir / "wait_specialist_feature_importance.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "cached_fusion_risk_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"key": "meta_train_rows", "value": len(train)},
            {"key": "holdout_rows", "value": len(holdout)},
            {"key": "meta_train_days", "value": args.meta_train_days},
            {"key": "device", "value": str(device)},
            {"key": "source_models", "value": "gcn,tcn_gcn,wait_specialist_for_wait_targets"},
        ]
    ).to_csv(args.output_dir / "cached_fusion_run_config.csv", index=False, encoding="utf-8-sig")
    print("Cached dynamic fusion completed.")
    print(metrics.to_string(index=False))
    print(weights.to_string(index=False))


if __name__ == "__main__":
    main()

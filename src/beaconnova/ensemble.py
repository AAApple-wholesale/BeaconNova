from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class FusionTrainConfig:
    epochs: int = 260
    batch_size: int = 2048
    hidden_dim: int = 48
    dropout: float = 0.08
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 24
    random_state: int = 42
    device: str = "auto"
    correction_weight: float = 0.001


@dataclass
class TrainedFusionTarget:
    target_name: str
    design_names: list[str]
    input_mean: np.ndarray
    input_std: np.ndarray
    model: "FusionGate"
    history: pd.DataFrame


@dataclass
class PredictionEnsemble:
    target_names: list[str]
    source_names: list[str]
    targets: dict[str, TrainedFusionTarget]
    config: FusionTrainConfig
    device: str


class FusionGate(nn.Module):
    def __init__(self, input_dim: int, source_count: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.weight_head = nn.Linear(hidden_dim, source_count)
        self.bias_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, design: torch.Tensor, raw_sources: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(design)
        weights = torch.softmax(self.weight_head(hidden), dim=1)
        bias = self.bias_head(hidden).squeeze(1)
        pred = torch.sum(weights * raw_sources, dim=1) + bias
        return pred, weights, bias


def _select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _candidate_features(target_name: str) -> list[str]:
    person_features = [
        "total_person_count",
        "total_person_count_lag_1",
        "total_person_count_lag_3",
        "total_person_count_lag_6",
        "total_person_count_lag_12",
        "total_person_count_lag_24",
        "total_person_count_roll_mean_3",
        "total_person_count_roll_mean_6",
        "total_person_count_roll_mean_12",
        "total_person_count_roll_mean_24",
        "total_person_count_roll_max_12",
        "total_person_count_roll_std_12",
        "person_diff_1",
        "person_diff_3",
        "person_diff_6",
        "person_momentum_30min",
        "person_momentum_60min",
        "gate_person_count",
        "system_person_count",
        "channel_load_share",
        "gate_system_share",
        "rail_passengers_30min",
        "rail_passengers_60min",
        "rail_passengers_120min",
        "rail_passengers_diff_30min",
        "ticket_count_30min",
        "ticket_count_60min",
        "ticket_count_120min",
        "ticket_count_diff_30min",
        "facility_hourly_pressure",
        "ticket_facility_hourly_pressure",
        "ticket_facility_120min_pressure",
    ]
    wait_features = [
        "avg_queue_wait_min",
        "avg_queue_wait_min_lag_1",
        "avg_queue_wait_min_lag_3",
        "avg_queue_wait_min_lag_6",
        "avg_queue_wait_min_lag_12",
        "avg_queue_wait_min_lag_24",
        "avg_queue_wait_min_roll_mean_3",
        "avg_queue_wait_min_roll_mean_6",
        "avg_queue_wait_min_roll_mean_12",
        "avg_queue_wait_min_roll_mean_24",
        "avg_queue_wait_min_roll_max_12",
        "avg_queue_wait_min_roll_std_12",
        "wait_diff_1",
        "wait_diff_3",
        "wait_diff_6",
        "wait_momentum_30min",
        "wait_momentum_60min",
        "gate_wait_mean",
        "system_wait_mean",
        "channel_wait_gap",
        "gate_wait_gap",
        "gate_flow_intensity",
        "bag_per_person",
        "wait_per_person",
        "rail_ropeway_hourly_pressure",
        "rail_ropeway_120min_pressure",
        "rail_ropeway_platform_pressure",
        "weather_rain_flag",
        "weather_rain_3h",
        "weather_rain_6h",
        "weather_wind_stress",
        "weather_comfort_penalty",
    ]
    shared = [
        "hour",
        "slot_5min",
        "sin_slot",
        "cos_slot",
        "dayofweek",
        "is_effective_weekend",
        "is_holiday",
        "is_summer_vacation",
        "days_to_holiday",
        "days_since_holiday",
    ]
    return (person_features if target_name.startswith("person") else wait_features) + shared


def _design_matrix(
    source_predictions: dict[str, np.ndarray],
    target_idx: int,
    x: np.ndarray,
    feature_names: list[str],
    security_indices: list[int],
    target_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    source_cols = []
    names = []
    for source_name, pred in source_predictions.items():
        source_cols.append(pred[:, :, target_idx].reshape(-1))
        names.append(f"pred_{source_name}")
    raw_sources = np.column_stack(source_cols).astype(np.float32)
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    security_x = x[:, security_indices, :]
    feature_cols = []
    for feature in _candidate_features(target_name):
        idx = feature_index.get(feature)
        if idx is None:
            continue
        names.append(feature)
        feature_cols.append(security_x[:, :, idx].reshape(-1))
    design = np.column_stack(source_cols + feature_cols).astype(np.float32)
    design = np.nan_to_num(design, nan=0.0, posinf=0.0, neginf=0.0)
    raw_sources = np.nan_to_num(raw_sources, nan=0.0, posinf=0.0, neginf=0.0)
    return design, raw_sources, names


def _fit_one_target(
    target_name: str,
    design: np.ndarray,
    raw_sources: np.ndarray,
    target: np.ndarray,
    source_count: int,
    config: FusionTrainConfig,
    device: torch.device,
) -> TrainedFusionTarget:
    mask = np.isfinite(target)
    design = design[mask]
    raw_sources = raw_sources[mask]
    target = target[mask].astype(np.float32)
    split = max(1, int(len(target) * 0.82))
    if split >= len(target):
        split = max(1, len(target) - 1)
    mean = design[:split].mean(axis=0)
    std = design[:split].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    design = ((design - mean) / std).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(design[:split]),
        torch.from_numpy(raw_sources[:split]),
        torch.from_numpy(target[:split]),
    )
    val_design = torch.from_numpy(design[split:]).to(device)
    val_sources = torch.from_numpy(raw_sources[split:]).to(device)
    val_target = torch.from_numpy(target[split:]).to(device)
    loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    model = FusionGate(design.shape[1], source_count, config.hidden_dim, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    best_state = None
    best_val = float("inf")
    stale = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for batch_design, batch_sources, batch_target in loader:
            batch_design = batch_design.to(device)
            batch_sources = batch_sources.to(device)
            batch_target = batch_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred, _, bias = model(batch_design, batch_sources)
            loss = loss_fn(pred, batch_target) + config.correction_weight * torch.mean(bias.square())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_pred, _, val_bias = model(val_design, val_sources)
            val_loss = loss_fn(val_pred, val_target) + config.correction_weight * torch.mean(val_bias.square())
            val_mae = torch.mean(torch.abs(torch.clamp(val_pred, min=0.0) - val_target))
        history.append(
            {
                "target": target_name,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "val_loss": float(val_loss.detach().cpu()),
                "val_mae": float(val_mae.detach().cpu()),
                "device": str(device),
            }
        )
        current = float(val_loss.detach().cpu())
        if current < best_val - 1e-5:
            best_val = current
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainedFusionTarget(target_name, [], mean.astype(np.float32), std.astype(np.float32), model, pd.DataFrame(history))


def fit_prediction_ensemble(
    source_predictions: dict[str, np.ndarray],
    val_y: np.ndarray,
    val_x: np.ndarray,
    feature_names: list[str],
    target_names: list[str],
    security_indices: list[int],
    config: FusionTrainConfig | None = None,
) -> PredictionEnsemble:
    if config is None:
        config = FusionTrainConfig()
    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)
    device = _select_device(config.device)
    targets: dict[str, TrainedFusionTarget] = {}
    source_count = len(source_predictions)
    for target_idx, target_name in enumerate(target_names):
        design, raw_sources, names = _design_matrix(source_predictions, target_idx, val_x, feature_names, security_indices, target_name)
        target = val_y[:, :, target_idx].reshape(-1)
        trained = _fit_one_target(target_name, design, raw_sources, target, source_count, config, device)
        trained.design_names = names
        targets[target_name] = trained
    return PredictionEnsemble(list(target_names), list(source_predictions.keys()), targets, config, str(device))


def apply_prediction_ensemble(
    ensemble: PredictionEnsemble,
    source_predictions: dict[str, np.ndarray],
    x: np.ndarray,
    feature_names: list[str],
    security_indices: list[int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    first = next(iter(source_predictions.values()))
    out = np.zeros_like(first, dtype=np.float32)
    weights_by_target: dict[str, np.ndarray] = {}
    device = torch.device(ensemble.device)
    for target_idx, target_name in enumerate(ensemble.target_names):
        target_model = ensemble.targets[target_name]
        design, raw_sources, _ = _design_matrix(source_predictions, target_idx, x, feature_names, security_indices, target_name)
        design = ((design - target_model.input_mean) / target_model.input_std).astype(np.float32)
        preds = []
        weights = []
        target_model.model.eval()
        with torch.no_grad():
            for start in range(0, len(design), max(1024, ensemble.config.batch_size * 2)):
                batch_design = torch.from_numpy(design[start : start + max(1024, ensemble.config.batch_size * 2)]).to(device)
                batch_sources = torch.from_numpy(raw_sources[start : start + max(1024, ensemble.config.batch_size * 2)]).to(device)
                pred, weight, _ = target_model.model(batch_design, batch_sources)
                preds.append(pred.detach().cpu().numpy())
                weights.append(weight.detach().cpu().numpy())
        out[:, :, target_idx] = np.concatenate(preds).reshape(first.shape[0], first.shape[1])
        weights_by_target[target_name] = np.concatenate(weights).reshape(first.shape[0], first.shape[1], len(ensemble.source_names))
    return np.clip(out, a_min=0.0, a_max=None), weights_by_target


def ensemble_frame(ensemble: PredictionEnsemble, weights_by_target: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    rows = []
    for target_name, target_model in ensemble.targets.items():
        if weights_by_target and target_name in weights_by_target:
            means = weights_by_target[target_name].reshape(-1, len(ensemble.source_names)).mean(axis=0)
        else:
            means = np.full(len(ensemble.source_names), np.nan)
        for source_name, weight in zip(ensemble.source_names, means):
            rows.append(
                {
                    "target": target_name,
                    "fusion_item": f"weight_{source_name}",
                    "mean_value": float(weight) if np.isfinite(weight) else np.nan,
                    "fusion_type": "dynamic_softmax_weight",
                    "epochs_trained": int(len(target_model.history)),
                }
            )
        rows.append(
            {
                "target": target_name,
                "fusion_item": "input_features",
                "mean_value": len(target_model.design_names),
                "fusion_type": "meta_learner_design_width",
                "epochs_trained": int(len(target_model.history)),
            }
        )
    return pd.DataFrame(rows)


def fusion_history_frame(ensemble: PredictionEnsemble) -> pd.DataFrame:
    frames = [target.history for target in ensemble.targets.values() if not target.history.empty]
    if not frames:
        return pd.DataFrame(columns=["target", "epoch", "train_loss", "val_loss", "val_mae", "device"])
    return pd.concat(frames, ignore_index=True)

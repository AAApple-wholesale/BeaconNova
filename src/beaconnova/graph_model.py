from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class GraphTrainConfig:
    epochs: int = 16
    batch_size: int = 512
    hidden_dim: int = 96
    dropout: float = 0.12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    random_state: int = 42
    patience: int = 5
    device: str = "auto"


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        propagated = torch.einsum("ij,bjf->bif", adj, x)
        return self.linear(propagated)


class ScenicGCN(nn.Module):
    """Small dense-adjacency GCN for the scenic-area operational graph."""

    def __init__(self, in_features: int, out_features: int, hidden_dim: int = 96, dropout: float = 0.12) -> None:
        super().__init__()
        self.input = nn.Linear(in_features, hidden_dim)
        self.gcn1 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.input(x))
        h1 = torch.relu(self.norm1(self.gcn1(h, adj)))
        h = self.dropout(h + h1)
        h2 = torch.relu(self.norm2(self.gcn2(h, adj)))
        h = self.dropout(h + h2)
        return self.head(h)


@dataclass
class TrainedGraphModel:
    model: ScenicGCN
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    train_config: GraphTrainConfig
    history: pd.DataFrame
    device: str


def _select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _standardize_features(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_values = x[train_idx].reshape(-1, x.shape[-1])
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x_scaled = (x - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    return x_scaled.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _standardize_targets(y: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_values = y[train_idx].reshape(-1, y.shape[-1])
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    y_scaled = (y - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    return y_scaled.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def train_graph_model(
    x: np.ndarray,
    y: np.ndarray,
    adjacency: np.ndarray,
    security_indices: list[int],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    config: GraphTrainConfig,
) -> TrainedGraphModel:
    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)
    device = _select_device(config.device)
    x_scaled, feature_mean, feature_std = _standardize_features(x, train_idx)
    y_scaled, target_mean, target_std = _standardize_targets(y, train_idx)

    train_ds = TensorDataset(torch.from_numpy(x_scaled[train_idx]), torch.from_numpy(y_scaled[train_idx]))
    val_x = torch.from_numpy(x_scaled[val_idx]).to(device)
    val_y = torch.from_numpy(y_scaled[val_idx]).to(device)
    loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    adj = torch.from_numpy(adjacency.astype(np.float32)).to(device)
    security_tensor = torch.tensor(security_indices, dtype=torch.long, device=device)

    model = ScenicGCN(x.shape[-1], y.shape[-1], hidden_dim=config.hidden_dim, dropout=config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    history = []
    best_state = None
    best_val = float("inf")
    stale = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x, adj).index_select(1, security_tensor)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x, adj).index_select(1, security_tensor)
            val_loss = float(loss_fn(val_pred, val_y).detach().cpu())
        train_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "device": str(device)})
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainedGraphModel(
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
        train_config=config,
        history=pd.DataFrame(history),
        device=str(device),
    )


def predict_graph_model(
    trained: TrainedGraphModel,
    x: np.ndarray,
    adjacency: np.ndarray,
    security_indices: list[int],
    batch_size: int = 1024,
) -> np.ndarray:
    device = torch.device(trained.device)
    x_scaled = (x - trained.feature_mean.reshape(1, 1, -1)) / trained.feature_std.reshape(1, 1, -1)
    x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    adj = torch.from_numpy(adjacency.astype(np.float32)).to(device)
    security_tensor = torch.tensor(security_indices, dtype=torch.long, device=device)
    preds = []
    trained.model.eval()
    with torch.no_grad():
        for start in range(0, len(x_scaled), batch_size):
            batch = torch.from_numpy(x_scaled[start : start + batch_size]).to(device)
            pred = trained.model(batch, adj).index_select(1, security_tensor)
            pred = pred.detach().cpu().numpy()
            preds.append(pred)
    scaled = np.concatenate(preds, axis=0)
    values = scaled * trained.target_std.reshape(1, 1, -1) + trained.target_mean.reshape(1, 1, -1)
    return np.clip(values, a_min=0.0, a_max=None)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TemporalGraphTrainConfig:
    epochs: int = 18
    batch_size: int = 384
    hidden_dim: int = 96
    sequence_length: int = 12
    dropout: float = 0.12
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    random_state: int = 42
    patience: int = 5
    device: str = "auto"
    adaptive_adj_rank: int = 8
    adaptive_adj_weight: float = 0.10


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.einsum("ij,bjf->bif", adj, x))


class GatedTemporalBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.conv(x)[..., -x.size(-1) :]
        gate, value = h.chunk(2, dim=1)
        h = torch.tanh(value) * torch.sigmoid(gate)
        h = self.proj(h)
        h = self.dropout(h)
        return torch.relu(self.norm(h + residual))


class TemporalGraphNetwork(nn.Module):
    """Gated dilated TCN + adaptive-adjacency GCN for scenic-flow forecasting."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = 96,
        dropout: float = 0.12,
        node_count: int = 1,
        adaptive_adj_rank: int = 8,
        adaptive_adj_weight: float = 0.10,
    ) -> None:
        super().__init__()
        rank = max(1, min(adaptive_adj_rank, node_count))
        self.adaptive_adj_weight = float(adaptive_adj_weight)
        self.input = nn.Linear(in_features, hidden_dim)
        self.temporal_blocks = nn.Sequential(
            GatedTemporalBlock(hidden_dim, kernel_size=3, dilation=1, dropout=dropout),
            GatedTemporalBlock(hidden_dim, kernel_size=3, dilation=2, dropout=dropout),
            GatedTemporalBlock(hidden_dim, kernel_size=3, dilation=4, dropout=dropout),
        )
        self.node_emb_source = nn.Parameter(torch.randn(node_count, rank) * 0.02)
        self.node_emb_target = nn.Parameter(torch.randn(rank, node_count) * 0.02)
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

    def _adjacency(self, adj: torch.Tensor) -> torch.Tensor:
        adaptive = torch.softmax(torch.relu(self.node_emb_source @ self.node_emb_target), dim=1)
        return (1.0 - self.adaptive_adj_weight) * adj + self.adaptive_adj_weight * adaptive

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch, seq, nodes, _ = x.shape
        h = torch.relu(self.input(x))
        h = h.permute(0, 2, 3, 1).reshape(batch * nodes, h.shape[-1], seq)
        h = self.temporal_blocks(h)
        h = (0.7 * h[..., -1] + 0.3 * h.mean(dim=-1)).reshape(batch, nodes, -1)
        adj = self._adjacency(adj)
        h1 = torch.relu(self.norm1(self.gcn1(h, adj)))
        h = self.dropout(h + h1)
        h2 = torch.relu(self.norm2(self.gcn2(h, adj)))
        h = self.dropout(h + h2)
        return self.head(h)


@dataclass
class TrainedTemporalGraphModel:
    model: TemporalGraphNetwork
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    config: TemporalGraphTrainConfig
    history: pd.DataFrame
    device: str


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_temporal_samples(x: np.ndarray, y: np.ndarray, sample_idx: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq_x = []
    seq_y = []
    end_idx = []
    valid = set(int(i) for i in sample_idx)
    for idx in sample_idx:
        idx = int(idx)
        start = idx - sequence_length + 1
        if start < 0:
            continue
        window = list(range(start, idx + 1))
        if not all(i in valid for i in window):
            continue
        seq_x.append(x[window])
        seq_y.append(y[idx])
        end_idx.append(idx)
    return np.asarray(seq_x, dtype=np.float32), np.asarray(seq_y, dtype=np.float32), np.asarray(end_idx, dtype=np.int64)


def _standardize(x: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None or std is None:
        flat = x.reshape(-1, x.shape[-1])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
    scaled = (x - mean.reshape((1,) * (x.ndim - 1) + (-1,))) / std.reshape((1,) * (x.ndim - 1) + (-1,))
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _standardize_targets(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = y.reshape(-1, y.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    scaled = (y - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    return scaled.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def train_temporal_graph_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    adjacency: np.ndarray,
    security_indices: list[int],
    config: TemporalGraphTrainConfig,
) -> TrainedTemporalGraphModel:
    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)
    device = _device(config.device)
    train_x, feature_mean, feature_std = _standardize(train_x)
    val_x, _, _ = _standardize(val_x, feature_mean, feature_std)
    train_y, target_mean, target_std = _standardize_targets(train_y)
    val_y = ((val_y - target_mean.reshape(1, 1, -1)) / target_std.reshape(1, 1, -1)).astype(np.float32)

    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=config.batch_size, shuffle=True)
    val_x_t = torch.from_numpy(val_x).to(device)
    val_y_t = torch.from_numpy(val_y).to(device)
    adj = torch.from_numpy(adjacency.astype(np.float32)).to(device)
    security_tensor = torch.tensor(security_indices, dtype=torch.long, device=device)
    model = TemporalGraphNetwork(
        train_x.shape[-1],
        train_y.shape[-1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        node_count=train_x.shape[2],
        adaptive_adj_rank=config.adaptive_adj_rank,
        adaptive_adj_weight=config.adaptive_adj_weight,
    ).to(device)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x_t, adj).index_select(1, security_tensor)
            val_loss = float(loss_fn(val_pred, val_y_t).detach().cpu())
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
    return TrainedTemporalGraphModel(model, feature_mean, feature_std, target_mean, target_std, config, pd.DataFrame(history), str(device))


def predict_temporal_graph_model(
    trained: TrainedTemporalGraphModel,
    x: np.ndarray,
    adjacency: np.ndarray,
    security_indices: list[int],
    batch_size: int = 512,
) -> np.ndarray:
    x, _, _ = _standardize(x, trained.feature_mean, trained.feature_std)
    device = torch.device(trained.device)
    adj = torch.from_numpy(adjacency.astype(np.float32)).to(device)
    security_tensor = torch.tensor(security_indices, dtype=torch.long, device=device)
    preds = []
    trained.model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            pred = trained.model(batch, adj).index_select(1, security_tensor).detach().cpu().numpy()
            preds.append(pred)
    scaled = np.concatenate(preds, axis=0)
    values = scaled * trained.target_std.reshape(1, 1, -1) + trained.target_mean.reshape(1, 1, -1)
    return np.clip(values, a_min=0.0, a_max=None)

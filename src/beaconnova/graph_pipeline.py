from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .baseline import regression_metrics
from .config import ModelConfig
from .data import load_context_assets, load_facility_capacity, load_rail_data, load_ropeway_capacity, load_security_data, load_ticket_aggregates
from .features import build_feature_frame
from .graph import build_graph_dataset, graph_edges_frame, graph_nodes_frame, split_graph_dataset
from .graph_model import GraphTrainConfig, predict_graph_model, train_graph_model
from .scoring import add_comfort_and_risk, risk_summary


@dataclass(frozen=True)
class GraphPipelineConfig:
    model: ModelConfig
    train: GraphTrainConfig = GraphTrainConfig()
    validation_days: int = 7


def _target_columns(horizons: tuple[int, ...]) -> list[str]:
    cols = []
    for horizon in horizons:
        cols.extend([f"target_person_h{horizon}", f"target_wait_h{horizon}"])
    return cols


def _prediction_columns(horizons: tuple[int, ...]) -> list[str]:
    cols = []
    for horizon in horizons:
        cols.extend([f"pred_person_h{horizon}", f"pred_wait_h{horizon}"])
    return cols


def _validation_split(train_idx: np.ndarray, times: pd.Series, validation_days: int) -> tuple[np.ndarray, np.ndarray]:
    train_times = times.iloc[train_idx]
    unique_dates = pd.Series(train_times.dt.normalize().unique()).sort_values()
    if len(unique_dates) <= validation_days + 2:
        cut = max(1, int(len(train_idx) * 0.85))
        return train_idx[:cut], train_idx[cut:]
    val_start = unique_dates.iloc[-validation_days]
    fit_idx = train_idx[times.iloc[train_idx].to_numpy() < val_start.to_datetime64()]
    val_idx = train_idx[times.iloc[train_idx].to_numpy() >= val_start.to_datetime64()]
    if len(fit_idx) == 0 or len(val_idx) == 0:
        cut = max(1, int(len(train_idx) * 0.85))
        return train_idx[:cut], train_idx[cut:]
    return fit_idx, val_idx


def _build_prediction_frame(dataset, sample_idx: np.ndarray, predicted: np.ndarray, horizons: tuple[int, ...]) -> pd.DataFrame:
    pred_cols = _prediction_columns(horizons)
    observed = dataset.observed[dataset.observed["datetime"].isin(dataset.times.iloc[sample_idx])].copy()
    observed = observed.sort_values(["datetime", "node_id"]).reset_index(drop=True)
    flat_pred = predicted.reshape(-1, predicted.shape[-1])
    if len(observed) != len(flat_pred):
        raise RuntimeError(f"Prediction rows {len(flat_pred)} do not match observed rows {len(observed)}")
    out = observed[["datetime", "security_gate", "channel", "node_id"]].copy()
    for idx, col in enumerate(pred_cols):
        out[col] = flat_pred[:, idx]
    return out


def run_graph_pipeline(config: GraphPipelineConfig) -> dict[str, Path]:
    output_dir = config.model.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    security = load_security_data(config.model.data_dir)
    rail = load_rail_data(config.model.data_dir)
    ticket = load_ticket_aggregates(config.model.data_dir)
    facility = load_facility_capacity(config.model.data_dir)
    ropeway = load_ropeway_capacity(config.model.data_dir)
    context_assets = load_context_assets(config.model.data_dir)
    frame = build_feature_frame(
        security,
        rail,
        config.model.horizons,
        ticket_df=ticket,
        facility_df=facility,
        ropeway_df=ropeway,
    )
    dataset = build_graph_dataset(frame, facility, ropeway, config.model.horizons, context_df=context_assets)
    train_all_idx, test_idx = split_graph_dataset(dataset, config.model.test_days)
    fit_idx, val_idx = _validation_split(train_all_idx, dataset.times, config.validation_days)

    trained = train_graph_model(
        dataset.x,
        dataset.y,
        dataset.graph.adjacency,
        dataset.graph.security_indices,
        fit_idx,
        val_idx,
        config.train,
    )
    test_pred = predict_graph_model(
        trained,
        dataset.x[test_idx],
        dataset.graph.adjacency,
        dataset.graph.security_indices,
        batch_size=max(config.train.batch_size, 512),
    )
    predictions = _build_prediction_frame(dataset, test_idx, test_pred, config.model.horizons)
    observed = dataset.observed[dataset.observed["datetime"].isin(dataset.times.iloc[test_idx])].copy()
    observed = observed.sort_values(["datetime", "node_id"]).reset_index(drop=True)
    scored = add_comfort_and_risk(predictions, observed, config.model.horizons)

    metrics_rows = []
    target_cols = _target_columns(config.model.horizons)
    pred_cols = _prediction_columns(config.model.horizons)
    for target, pred_col in zip(target_cols, pred_cols):
        metric = regression_metrics(observed[target], scored[pred_col])
        horizon = int(target.rsplit("_h", 1)[-1]) * 5
        metric.update({"target": target, "horizon_min": horizon, "rows": int(observed[target].notna().sum())})
        metrics_rows.append(metric)
    metrics = pd.DataFrame(metrics_rows)
    summary = risk_summary(scored, config.model.horizons)

    prediction_path = output_dir / "graph_predictions.csv"
    metrics_path = output_dir / "graph_metrics.csv"
    summary_path = output_dir / "graph_risk_summary.csv"
    nodes_path = output_dir / "graph_nodes.csv"
    edges_path = output_dir / "graph_edges.csv"
    feature_manifest_path = output_dir / "graph_feature_manifest.csv"
    history_path = output_dir / "graph_training_history.csv"
    model_path = output_dir / "graph_model.pt"
    config_path = output_dir / "graph_run_config.csv"
    context_assets_path = output_dir / "context_assets.csv"

    scored.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    graph_nodes_frame(dataset.graph).to_csv(nodes_path, index=False, encoding="utf-8-sig")
    graph_edges_frame(dataset.graph).to_csv(edges_path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": dataset.feature_names}).to_csv(feature_manifest_path, index=False, encoding="utf-8-sig")
    trained.history.to_csv(history_path, index=False, encoding="utf-8-sig")
    context_assets.to_csv(context_assets_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"key": "samples", "value": len(dataset.times)},
            {"key": "fit_samples", "value": len(fit_idx)},
            {"key": "validation_samples", "value": len(val_idx)},
            {"key": "test_samples", "value": len(test_idx)},
            {"key": "nodes", "value": len(dataset.graph.node_ids)},
            {"key": "edges_directed", "value": len(graph_edges_frame(dataset.graph))},
            {"key": "features", "value": len(dataset.feature_names)},
            {"key": "device", "value": trained.device},
            {"key": "train_config", "value": asdict(config.train)},
        ]
    ).to_csv(config_path, index=False, encoding="utf-8-sig")
    torch.save(
        {
            "model_state_dict": trained.model.state_dict(),
            "feature_mean": trained.feature_mean,
            "feature_std": trained.feature_std,
            "target_mean": trained.target_mean,
            "target_std": trained.target_std,
            "feature_names": dataset.feature_names,
            "target_names": dataset.target_names,
            "node_ids": dataset.graph.node_ids,
            "security_indices": dataset.graph.security_indices,
            "adjacency": dataset.graph.adjacency,
            "train_config": asdict(config.train),
        },
        model_path,
    )

    return {
        "predictions": prediction_path,
        "metrics": metrics_path,
        "risk_summary": summary_path,
        "graph_nodes": nodes_path,
        "graph_edges": edges_path,
        "feature_manifest": feature_manifest_path,
        "training_history": history_path,
        "model": model_path,
        "run_config": config_path,
        "context_assets": context_assets_path,
    }

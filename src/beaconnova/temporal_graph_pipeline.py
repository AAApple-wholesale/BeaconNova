from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .baseline import regression_metrics
from .calibration import apply_prediction_calibrator, calibration_frame, fit_prediction_calibrator
from .config import ModelConfig
from .data import load_context_assets, load_facility_capacity, load_rail_data, load_ropeway_capacity, load_security_data, load_ticket_aggregates, load_weather_context
from .features import build_feature_frame
from .graph import build_graph_dataset, graph_edges_frame, graph_nodes_frame, split_graph_dataset
from .scoring import add_comfort_and_risk, risk_summary
from .temporal_graph_model import TemporalGraphTrainConfig, make_temporal_samples, predict_temporal_graph_model, train_temporal_graph_model
from .topology import calibrate_graph_edges


@dataclass(frozen=True)
class TemporalGraphPipelineConfig:
    model: ModelConfig
    train: TemporalGraphTrainConfig = TemporalGraphTrainConfig()
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


def _prediction_frame(dataset, end_idx: np.ndarray, predicted: np.ndarray, horizons: tuple[int, ...]) -> pd.DataFrame:
    pred_cols = _prediction_columns(horizons)
    observed = dataset.observed[dataset.observed["datetime"].isin(dataset.times.iloc[end_idx])].copy()
    observed = observed.sort_values(["datetime", "node_id"]).reset_index(drop=True)
    flat_pred = predicted.reshape(-1, predicted.shape[-1])
    if len(observed) != len(flat_pred):
        raise RuntimeError(f"Prediction rows {len(flat_pred)} do not match observed rows {len(observed)}")
    out = observed[["datetime", "security_gate", "channel", "node_id"]].copy()
    for idx, col in enumerate(pred_cols):
        out[col] = flat_pred[:, idx]
    return out


def run_temporal_graph_pipeline(config: TemporalGraphPipelineConfig) -> dict[str, Path]:
    output_dir = config.model.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    security = load_security_data(config.model.data_dir)
    rail = load_rail_data(config.model.data_dir)
    ticket = load_ticket_aggregates(config.model.data_dir)
    facility = load_facility_capacity(config.model.data_dir)
    ropeway = load_ropeway_capacity(config.model.data_dir)
    context_assets = load_context_assets(config.model.data_dir)
    weather = load_weather_context(config.model.data_dir, security["datetime"]) if config.model.use_weather_features else None
    frame = build_feature_frame(
        security,
        rail,
        config.model.horizons,
        ticket_df=ticket,
        facility_df=facility,
        ropeway_df=ropeway,
        weather_df=weather,
    )
    dataset = build_graph_dataset(frame, facility, ropeway, config.model.horizons, context_df=context_assets)
    adjacency, edge_weights = calibrate_graph_edges(frame, dataset.graph)
    train_all_idx, test_idx = split_graph_dataset(dataset, config.model.test_days)
    fit_idx, val_idx = _validation_split(train_all_idx, dataset.times, config.validation_days)

    train_x, train_y, train_end = make_temporal_samples(dataset.x, dataset.y, fit_idx, config.train.sequence_length)
    val_x, val_y, val_end = make_temporal_samples(dataset.x, dataset.y, val_idx, config.train.sequence_length)
    test_x, _, test_end = make_temporal_samples(dataset.x, dataset.y, test_idx, config.train.sequence_length)
    if len(train_x) == 0 or len(val_x) == 0 or len(test_x) == 0:
        raise RuntimeError("Not enough continuous samples for temporal graph training")

    trained = train_temporal_graph_model(train_x, train_y, val_x, val_y, adjacency, dataset.graph.security_indices, config.train)
    val_pred = predict_temporal_graph_model(trained, val_x, adjacency, dataset.graph.security_indices, batch_size=max(256, config.train.batch_size))
    calibrator = fit_prediction_calibrator(
        val_pred,
        val_y,
        dataset.x[val_end],
        dataset.feature_names,
        dataset.target_names,
        dataset.graph.security_indices,
    )
    test_pred = predict_temporal_graph_model(trained, test_x, adjacency, dataset.graph.security_indices, batch_size=max(256, config.train.batch_size))
    test_pred = apply_prediction_calibrator(
        calibrator,
        test_pred,
        dataset.x[test_end],
        dataset.feature_names,
        dataset.graph.security_indices,
    )
    predictions = _prediction_frame(dataset, test_end, test_pred, config.model.horizons)
    observed = dataset.observed[dataset.observed["datetime"].isin(dataset.times.iloc[test_end])].copy()
    observed = observed.sort_values(["datetime", "node_id"]).reset_index(drop=True)
    scored = add_comfort_and_risk(predictions, observed, config.model.horizons)

    metrics_rows = []
    for target, pred_col in zip(_target_columns(config.model.horizons), _prediction_columns(config.model.horizons)):
        metric = regression_metrics(observed[target], scored[pred_col])
        horizon = int(target.rsplit("_h", 1)[-1]) * 5
        metric.update({"target": target, "horizon_min": horizon, "rows": int(observed[target].notna().sum())})
        metrics_rows.append(metric)
    metrics = pd.DataFrame(metrics_rows)
    summary = risk_summary(scored, config.model.horizons)

    paths = {
        "predictions": output_dir / "temporal_graph_predictions.csv",
        "metrics": output_dir / "temporal_graph_metrics.csv",
        "risk_summary": output_dir / "temporal_graph_risk_summary.csv",
        "graph_nodes": output_dir / "temporal_graph_nodes.csv",
        "graph_edges": output_dir / "temporal_graph_edges.csv",
        "edge_weights": output_dir / "temporal_graph_edge_weights.csv",
        "feature_manifest": output_dir / "temporal_graph_feature_manifest.csv",
        "training_history": output_dir / "temporal_graph_training_history.csv",
        "calibration": output_dir / "temporal_graph_calibration_coefficients.csv",
        "context_assets": output_dir / "context_assets.csv",
        "model": output_dir / "temporal_graph_model.pt",
        "run_config": output_dir / "temporal_graph_run_config.csv",
    }
    scored.to_csv(paths["predictions"], index=False, encoding="utf-8-sig")
    metrics.to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["risk_summary"], index=False, encoding="utf-8-sig")
    graph_nodes_frame(dataset.graph).to_csv(paths["graph_nodes"], index=False, encoding="utf-8-sig")
    graph_edges_frame(dataset.graph).to_csv(paths["graph_edges"], index=False, encoding="utf-8-sig")
    edge_weights.to_csv(paths["edge_weights"], index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": dataset.feature_names}).to_csv(paths["feature_manifest"], index=False, encoding="utf-8-sig")
    trained.history.to_csv(paths["training_history"], index=False, encoding="utf-8-sig")
    calibration_frame(calibrator).to_csv(paths["calibration"], index=False, encoding="utf-8-sig")
    context_assets.to_csv(paths["context_assets"], index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"key": "samples", "value": len(dataset.times)},
            {"key": "sequence_length", "value": config.train.sequence_length},
            {"key": "fit_sequences", "value": len(train_x)},
            {"key": "validation_sequences", "value": len(val_x)},
            {"key": "test_sequences", "value": len(test_x)},
            {"key": "nodes", "value": len(dataset.graph.node_ids)},
            {"key": "edges_directed", "value": len(graph_edges_frame(dataset.graph))},
            {"key": "edge_weight_mean", "value": float(edge_weights["calibrated_weight"].mean()) if not edge_weights.empty else 0.0},
            {"key": "prediction_calibration", "value": "validation_ridge_linear"},
            {"key": "features", "value": len(dataset.feature_names)},
            {"key": "device", "value": trained.device},
            {"key": "train_config", "value": asdict(config.train)},
        ]
    ).to_csv(paths["run_config"], index=False, encoding="utf-8-sig")
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
            "adjacency": adjacency,
            "prediction_calibrator": {
                "target_names": calibrator.target_names,
                "feature_names": calibrator.feature_names,
                "coefficients": {key: value.tolist() for key, value in calibrator.coefficients.items()},
                "ridge_alpha": calibrator.ridge_alpha,
            },
            "train_config": asdict(config.train),
        },
        paths["model"],
    )
    return paths





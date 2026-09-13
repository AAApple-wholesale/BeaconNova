from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import MultiTargetBaseline, regression_metrics
from .calibration import apply_prediction_calibrator, calibration_frame, fit_prediction_calibrator
from .config import ModelConfig
from .data import load_context_assets, load_facility_capacity, load_rail_data, load_ropeway_capacity, load_security_data, load_ticket_aggregates, load_weather_context
from .ensemble import FusionTrainConfig, apply_prediction_ensemble, ensemble_frame, fit_prediction_ensemble, fusion_history_frame
from .features import build_feature_frame, feature_columns
from .graph import build_graph_dataset, graph_edges_frame, graph_nodes_frame, split_graph_dataset
from .graph_model import GraphTrainConfig, predict_graph_model, train_graph_model
from .scoring import add_comfort_and_risk, risk_summary
from .temporal_graph_model import TemporalGraphTrainConfig, make_temporal_samples, predict_temporal_graph_model, train_temporal_graph_model
from .topology import calibrate_graph_edges


@dataclass(frozen=True)
class EnsemblePipelineConfig:
    model: ModelConfig
    graph_train: GraphTrainConfig = GraphTrainConfig(epochs=18, hidden_dim=128, dropout=0.10, adaptive_adj_weight=0.10)
    temporal_train: TemporalGraphTrainConfig = TemporalGraphTrainConfig(epochs=20, hidden_dim=128, dropout=0.10, sequence_length=18, adaptive_adj_weight=0.08)
    validation_days: int = 10
    fusion_train: FusionTrainConfig = FusionTrainConfig()
    include_tabular: bool = False


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


def _observed_for(dataset, sample_idx: np.ndarray) -> pd.DataFrame:
    observed = dataset.observed[dataset.observed["datetime"].isin(dataset.times.iloc[sample_idx])].copy()
    return observed.sort_values(["datetime", "node_id"]).reset_index(drop=True)


def _prediction_frame(dataset, sample_idx: np.ndarray, predicted: np.ndarray, horizons: tuple[int, ...]) -> pd.DataFrame:
    pred_cols = _prediction_columns(horizons)
    observed = _observed_for(dataset, sample_idx)
    flat_pred = predicted.reshape(-1, predicted.shape[-1])
    if len(observed) != len(flat_pred):
        raise RuntimeError(f"Prediction rows {len(flat_pred)} do not match observed rows {len(observed)}")
    out = observed[["datetime", "security_gate", "channel", "node_id"]].copy()
    for idx, col in enumerate(pred_cols):
        out[col] = flat_pred[:, idx]
    return out


def _tabular_prediction_tensor(model: MultiTargetBaseline, rows: pd.DataFrame, feature_cols: list[str], horizons: tuple[int, ...], sample_count: int, node_count: int) -> np.ndarray:
    pred_frame = model.predict(rows, feature_cols).sort_values(["datetime", "node_id"]).reset_index(drop=True)
    values = pred_frame[_prediction_columns(horizons)].to_numpy(dtype=np.float32)
    return values.reshape(sample_count, node_count, len(_prediction_columns(horizons)))


def _persistence_prediction_tensor(dataset, sample_idx: np.ndarray, horizons: tuple[int, ...]) -> np.ndarray:
    feature_index = {name: idx for idx, name in enumerate(dataset.feature_names)}
    security_x = dataset.x[sample_idx][:, dataset.graph.security_indices, :]
    cols = []
    for horizon in horizons:
        person_col = f"total_person_count_lag_{horizon}"
        wait_col = f"avg_queue_wait_min_lag_{horizon}"
        person_idx = feature_index.get(person_col, feature_index["total_person_count"])
        wait_idx = feature_index.get(wait_col, feature_index["avg_queue_wait_min"])
        cols.append(security_x[:, :, person_idx])
        cols.append(security_x[:, :, wait_idx])
    return np.stack(cols, axis=-1).astype(np.float32)


def _source_metrics(name: str, pred: np.ndarray, truth: np.ndarray, target_names: list[str], horizons: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for target_idx, target_name in enumerate(target_names):
        target_col = "target_" + target_name
        pred_series = pd.Series(pred[:, :, target_idx].reshape(-1))
        true_series = pd.Series(truth[:, :, target_idx].reshape(-1))
        metric = regression_metrics(true_series, pred_series)
        horizon = int(target_name.rsplit("_h", 1)[-1]) * 5
        metric.update({"source": name, "target": target_col, "horizon_min": horizon, "rows": int(true_series.notna().sum())})
        rows.append(metric)
    return pd.DataFrame(rows)


def run_ensemble_pipeline(config: EnsemblePipelineConfig) -> dict[str, Path]:
    output_dir = config.model.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    security = load_security_data(config.model.data_dir)
    rail = load_rail_data(config.model.data_dir)
    ticket = load_ticket_aggregates(config.model.data_dir) if config.model.use_ticket_features else None
    facility = load_facility_capacity(config.model.data_dir) if config.model.use_facility_features else None
    ropeway = load_ropeway_capacity(config.model.data_dir) if config.model.use_facility_features else None
    context_assets = load_context_assets(config.model.data_dir)
    weather = load_weather_context(config.model.data_dir, security["datetime"]) if config.model.use_weather_features else None
    frame = build_feature_frame(security, rail, config.model.horizons, ticket_df=ticket, facility_df=facility, ropeway_df=ropeway, weather_df=weather)
    dataset = build_graph_dataset(frame, facility, ropeway, config.model.horizons, context_df=context_assets)
    adjacency, edge_weights = calibrate_graph_edges(frame, dataset.graph)
    train_all_idx, test_idx = split_graph_dataset(dataset, config.model.test_days)
    fit_idx, val_idx = _validation_split(train_all_idx, dataset.times, config.validation_days)

    target_cols = _target_columns(config.model.horizons)
    feature_cols = feature_columns(frame)
    val_tabular = None
    test_tabular = None
    if config.include_tabular:
        fit_times = set(dataset.times.iloc[fit_idx])
        fit_rows = frame[frame["datetime"].isin(fit_times)].dropna(subset=target_cols).copy()
        val_rows = _observed_for(dataset, val_idx)
        test_rows = _observed_for(dataset, test_idx)
        tabular = MultiTargetBaseline(target_cols, random_state=config.model.random_state)
        tabular.fit(fit_rows, feature_cols)
        val_tabular = _tabular_prediction_tensor(tabular, val_rows, feature_cols, config.model.horizons, len(val_idx), len(dataset.graph.security_indices))
        test_tabular = _tabular_prediction_tensor(tabular, test_rows, feature_cols, config.model.horizons, len(test_idx), len(dataset.graph.security_indices))

    graph_trained = train_graph_model(dataset.x, dataset.y, adjacency, dataset.graph.security_indices, fit_idx, val_idx, config.graph_train)
    val_graph_raw = predict_graph_model(graph_trained, dataset.x[val_idx], adjacency, dataset.graph.security_indices, batch_size=max(config.graph_train.batch_size, 512))
    graph_calibrator = fit_prediction_calibrator(val_graph_raw, dataset.y[val_idx], dataset.x[val_idx], dataset.feature_names, dataset.target_names, dataset.graph.security_indices)
    val_graph = apply_prediction_calibrator(graph_calibrator, val_graph_raw, dataset.x[val_idx], dataset.feature_names, dataset.graph.security_indices)
    test_graph_raw = predict_graph_model(graph_trained, dataset.x[test_idx], adjacency, dataset.graph.security_indices, batch_size=max(config.graph_train.batch_size, 512))
    test_graph = apply_prediction_calibrator(graph_calibrator, test_graph_raw, dataset.x[test_idx], dataset.feature_names, dataset.graph.security_indices)

    train_x, train_y, _ = make_temporal_samples(dataset.x, dataset.y, fit_idx, config.temporal_train.sequence_length)
    temporal_val_x, temporal_val_y, temporal_val_end = make_temporal_samples(dataset.x, dataset.y, val_idx, config.temporal_train.sequence_length)
    temporal_test_x, _, temporal_test_end = make_temporal_samples(dataset.x, dataset.y, test_idx, config.temporal_train.sequence_length)
    if len(train_x) == 0 or len(temporal_val_x) == 0 or len(temporal_test_x) == 0:
        raise RuntimeError("Not enough continuous samples for temporal graph ensemble training")
    temporal_trained = train_temporal_graph_model(train_x, train_y, temporal_val_x, temporal_val_y, adjacency, dataset.graph.security_indices, config.temporal_train)
    val_temporal_raw = predict_temporal_graph_model(temporal_trained, temporal_val_x, adjacency, dataset.graph.security_indices, batch_size=max(256, config.temporal_train.batch_size))
    temporal_calibrator = fit_prediction_calibrator(val_temporal_raw, temporal_val_y, dataset.x[temporal_val_end], dataset.feature_names, dataset.target_names, dataset.graph.security_indices)
    val_temporal = apply_prediction_calibrator(temporal_calibrator, val_temporal_raw, dataset.x[temporal_val_end], dataset.feature_names, dataset.graph.security_indices)
    test_temporal_raw = predict_temporal_graph_model(temporal_trained, temporal_test_x, adjacency, dataset.graph.security_indices, batch_size=max(256, config.temporal_train.batch_size))
    test_temporal = apply_prediction_calibrator(temporal_calibrator, test_temporal_raw, dataset.x[temporal_test_end], dataset.feature_names, dataset.graph.security_indices)

    val_end_lookup = {int(idx): pos for pos, idx in enumerate(val_idx)}
    test_end_lookup = {int(idx): pos for pos, idx in enumerate(test_idx)}
    common_val_positions = np.asarray([val_end_lookup[int(idx)] for idx in temporal_val_end], dtype=np.int64)
    common_test_positions = np.asarray([test_end_lookup[int(idx)] for idx in temporal_test_end], dtype=np.int64)

    val_sources = {
        "gcn": val_graph[common_val_positions],
        "tcn_gcn": val_temporal,
        "persistence": _persistence_prediction_tensor(dataset, temporal_val_end, config.model.horizons),
    }
    test_sources = {
        "gcn": test_graph[common_test_positions],
        "tcn_gcn": test_temporal,
        "persistence": _persistence_prediction_tensor(dataset, temporal_test_end, config.model.horizons),
    }
    if config.include_tabular and val_tabular is not None and test_tabular is not None:
        val_sources["tabular_hgb"] = val_tabular[common_val_positions]
        test_sources["tabular_hgb"] = test_tabular[common_test_positions]
    ensemble = fit_prediction_ensemble(
        val_sources,
        dataset.y[temporal_val_end],
        dataset.x[temporal_val_end],
        dataset.feature_names,
        dataset.target_names,
        dataset.graph.security_indices,
        config.fusion_train,
    )
    test_pred, test_weights = apply_prediction_ensemble(
        ensemble,
        test_sources,
        dataset.x[temporal_test_end],
        dataset.feature_names,
        dataset.graph.security_indices,
    )
    predictions = _prediction_frame(dataset, temporal_test_end, test_pred, config.model.horizons)
    observed = _observed_for(dataset, temporal_test_end)
    scored = add_comfort_and_risk(predictions, observed, config.model.horizons)

    metrics_rows = []
    for target, pred_col in zip(target_cols, _prediction_columns(config.model.horizons)):
        metric = regression_metrics(observed[target], scored[pred_col])
        horizon = int(target.rsplit("_h", 1)[-1]) * 5
        metric.update({"target": target, "horizon_min": horizon, "rows": int(observed[target].notna().sum())})
        metrics_rows.append(metric)
    metrics = pd.DataFrame(metrics_rows)
    source_metrics = pd.concat(
        [
            _source_metrics("gcn", test_sources["gcn"], dataset.y[temporal_test_end], dataset.target_names, config.model.horizons),
            _source_metrics("tcn_gcn", test_sources["tcn_gcn"], dataset.y[temporal_test_end], dataset.target_names, config.model.horizons),
            *([_source_metrics("tabular_hgb", test_sources["tabular_hgb"], dataset.y[temporal_test_end], dataset.target_names, config.model.horizons)] if "tabular_hgb" in test_sources else []),
            _source_metrics("persistence", test_sources["persistence"], dataset.y[temporal_test_end], dataset.target_names, config.model.horizons),
            _source_metrics("ensemble", test_pred, dataset.y[temporal_test_end], dataset.target_names, config.model.horizons),
        ],
        ignore_index=True,
    )
    summary = risk_summary(scored, config.model.horizons)

    paths = {
        "predictions": output_dir / "ensemble_predictions.csv",
        "metrics": output_dir / "ensemble_metrics.csv",
        "source_metrics": output_dir / "ensemble_source_metrics.csv",
        "risk_summary": output_dir / "ensemble_risk_summary.csv",
        "coefficients": output_dir / "ensemble_coefficients.csv",
        "fusion_history": output_dir / "ensemble_fusion_training_history.csv",
        "feature_manifest": output_dir / "ensemble_feature_manifest.csv",
        "graph_nodes": output_dir / "ensemble_graph_nodes.csv",
        "graph_edges": output_dir / "ensemble_graph_edges.csv",
        "edge_weights": output_dir / "ensemble_edge_weights.csv",
        "graph_calibration": output_dir / "ensemble_graph_calibration_coefficients.csv",
        "temporal_calibration": output_dir / "ensemble_temporal_calibration_coefficients.csv",
        "graph_history": output_dir / "ensemble_graph_training_history.csv",
        "temporal_history": output_dir / "ensemble_temporal_training_history.csv",
        "context_assets": output_dir / "context_assets.csv",
        "run_config": output_dir / "ensemble_run_config.csv",
    }
    scored.to_csv(paths["predictions"], index=False, encoding="utf-8-sig")
    metrics.to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
    source_metrics.to_csv(paths["source_metrics"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["risk_summary"], index=False, encoding="utf-8-sig")
    ensemble_frame(ensemble, test_weights).to_csv(paths["coefficients"], index=False, encoding="utf-8-sig")
    fusion_history_frame(ensemble).to_csv(paths["fusion_history"], index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": dataset.feature_names}).to_csv(paths["feature_manifest"], index=False, encoding="utf-8-sig")
    graph_nodes_frame(dataset.graph).to_csv(paths["graph_nodes"], index=False, encoding="utf-8-sig")
    graph_edges_frame(dataset.graph).to_csv(paths["graph_edges"], index=False, encoding="utf-8-sig")
    edge_weights.to_csv(paths["edge_weights"], index=False, encoding="utf-8-sig")
    calibration_frame(graph_calibrator).to_csv(paths["graph_calibration"], index=False, encoding="utf-8-sig")
    calibration_frame(temporal_calibrator).to_csv(paths["temporal_calibration"], index=False, encoding="utf-8-sig")
    graph_trained.history.to_csv(paths["graph_history"], index=False, encoding="utf-8-sig")
    temporal_trained.history.to_csv(paths["temporal_history"], index=False, encoding="utf-8-sig")
    context_assets.to_csv(paths["context_assets"], index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"key": "samples", "value": len(dataset.times)},
            {"key": "fit_samples", "value": len(fit_idx)},
            {"key": "validation_samples", "value": len(val_idx)},
            {"key": "test_samples", "value": len(test_idx)},
            {"key": "ensemble_test_samples", "value": len(temporal_test_end)},
            {"key": "nodes", "value": len(dataset.graph.node_ids)},
            {"key": "features", "value": len(dataset.feature_names)},
            {"key": "sources", "value": ",".join(test_sources.keys())},
            {"key": "include_tabular", "value": config.include_tabular},
            {"key": "graph_train_config", "value": asdict(config.graph_train)},
            {"key": "temporal_train_config", "value": asdict(config.temporal_train)},
            {"key": "fusion_train_config", "value": asdict(config.fusion_train)},
        ]
    ).to_csv(paths["run_config"], index=False, encoding="utf-8-sig")
    return paths

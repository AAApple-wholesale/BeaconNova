from __future__ import annotations

from pathlib import Path

import pandas as pd

from .baseline import MultiTargetBaseline, chronological_split, regression_metrics
from .config import ModelConfig
from .data import load_rail_data, load_security_data, load_ticket_aggregates
from .features import build_feature_frame, feature_columns
from .scoring import add_comfort_and_risk, risk_summary


def run_baseline(config: ModelConfig) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    security = load_security_data(config.data_dir)
    rail = load_rail_data(config.data_dir)
    ticket = load_ticket_aggregates(config.data_dir) if config.use_ticket_features else None
    frame = build_feature_frame(security, rail, config.horizons, ticket_df=ticket)

    target_cols = []
    for horizon in config.horizons:
        target_cols.extend([f"target_person_h{horizon}", f"target_wait_h{horizon}"])

    model_frame = frame.dropna(subset=target_cols, how="all").copy()
    train, test = chronological_split(model_frame, config.test_days)
    cols = feature_columns(model_frame)

    model = MultiTargetBaseline(target_cols, random_state=config.random_state)
    model.fit(train, cols)
    predictions = model.predict(test, cols)
    scored = add_comfort_and_risk(predictions, test, config.horizons)

    metrics_rows = []
    for target in target_cols:
        pred_col = "pred_" + target.removeprefix("target_")
        metric = regression_metrics(test[target], scored[pred_col])
        horizon = int(target.rsplit("_h", 1)[-1]) * 5
        metric.update({"target": target, "horizon_min": horizon, "rows": int(test[target].notna().sum())})
        metrics_rows.append(metric)
    metrics = pd.DataFrame(metrics_rows)
    summary = risk_summary(scored, config.horizons)

    prediction_path = config.output_dir / "predictions.csv"
    metrics_path = config.output_dir / "metrics.csv"
    summary_path = config.output_dir / "risk_summary.csv"
    feature_manifest_path = config.output_dir / "feature_manifest.csv"
    scored.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": cols}).to_csv(feature_manifest_path, index=False, encoding="utf-8-sig")

    return {
        "predictions": prediction_path,
        "metrics": metrics_path,
        "risk_summary": summary_path,
        "feature_manifest": feature_manifest_path,
    }

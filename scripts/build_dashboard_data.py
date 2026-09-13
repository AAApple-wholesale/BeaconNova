"""Build compact JSON data for the local BeaconNova dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _round(value: Any, digits: int = 4) -> float | int | str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, str)):
        return value
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return str(value)


def _metric_block(name: str, output_dir: Path, prefix: str) -> dict[str, Any]:
    metrics = _read_csv(output_dir / f"{prefix}_metrics.csv")
    run_config = _read_csv(output_dir / f"{prefix}_run_config.csv")
    risk = _read_csv(output_dir / f"{prefix}_risk_summary.csv")

    metric_rows = []
    for _, row in metrics.iterrows():
        metric_rows.append(
            {
                "target": row.get("target"),
                "horizon_min": int(row.get("horizon_min", 0)),
                "mae": _round(row.get("mae")),
                "rmse": _round(row.get("rmse")),
                "wape": _round(row.get("wape")),
                "rows": int(row.get("rows", 0)),
            }
        )

    config = {
        str(row.get("key")): _round(row.get("value"))
        for _, row in run_config.iterrows()
        if "key" in row and "value" in row
    }

    risk_rows = []
    for _, row in risk.iterrows():
        risk_rows.append(
            {
                "horizon_min": int(row.get("horizon_min", 0)),
                "label": row.get("risk_level"),
                "count": int(row.get("count", 0)),
            }
        )

    return {
        "name": name,
        "path": str(output_dir),
        "metrics": metric_rows,
        "risk": risk_rows,
        "config": config,
    }


def _strategy_block(output_dir: Path) -> dict[str, Any]:
    summary = _read_csv(output_dir / "strategy_summary.csv")
    recs = _read_csv(output_dir / "strategy_recommendations.csv")
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "strategy": row.get("strategy_name"),
                "interventions": int(row.get("interventions", 0)),
                "avg_risk_score_delta": _round(row.get("avg_risk_score_delta")),
                "avg_comfort_gain": _round(row.get("avg_comfort_gain")),
                "avg_wait_reduction_min": _round(row.get("avg_wait_reduction_min")),
                "baseline_orange_red": int(row.get("baseline_orange_red", 0)),
                "adjusted_orange_red": int(row.get("adjusted_orange_red", 0)),
                "orange_red_reduction": int(row.get("orange_red_reduction", 0)),
            }
        )

    examples = []
    if not recs.empty:
        sort_col = "risk_score_delta"
        if sort_col not in recs.columns:
            sort_col = "baseline_risk_score"
        top = recs.sort_values(sort_col, ascending=False).head(8)
        for _, row in top.iterrows():
            examples.append(
                {
                    "datetime": row.get("datetime"),
                    "node_id": row.get("node_id"),
                    "strategy": row.get("strategy_name"),
                    "baseline_level": row.get("baseline_risk_level"),
                    "adjusted_level": row.get("adjusted_risk_level"),
                    "risk_delta": _round(row.get("risk_score_delta")),
                    "comfort_gain": _round(row.get("comfort_gain")),
                    "wait_reduction_min": _round(row.get("wait_reduction_min")),
                }
            )
    return {"summary": rows, "examples": examples}


def _weather_block(data_dir: Path) -> dict[str, Any]:
    weather = _read_csv(data_dir / "weather.csv")
    if weather.empty:
        return {}
    out = {
        "rows": int(len(weather)),
        "start": str(weather["datetime"].min()),
        "end": str(weather["datetime"].max()),
        "source": str(weather.get("weather_source", pd.Series(["unknown"])).iloc[0]),
    }
    for col in [
        "weather_temp_c",
        "weather_apparent_temp_c",
        "weather_humidity",
        "weather_rain_mm",
        "weather_wind_speed_mps",
        "weather_comfort_penalty",
    ]:
        if col in weather.columns:
            out[col] = {
                "mean": _round(weather[col].mean()),
                "max": _round(weather[col].max()),
                "min": _round(weather[col].min()),
            }
    return out


def _edge_block(output_dir: Path) -> list[dict[str, Any]]:
    edges = _read_csv(output_dir / "graph_edge_weights.csv")
    if edges.empty:
        edges = _read_csv(output_dir / "temporal_graph_edge_weights.csv")
    if edges.empty:
        return []
    top = edges.sort_values("calibrated_weight", ascending=False).head(12)
    return [
        {
            "source": row.get("source"),
            "target": row.get("target"),
            "relation": row.get("relation"),
            "prior_weight": _round(row.get("prior_weight")),
            "lag_corr": _round(row.get("lag_corr")),
            "best_lag_5min": int(row.get("best_lag_5min", 0)),
            "calibrated_weight": _round(row.get("calibrated_weight")),
        }
        for _, row in top.iterrows()
    ]


def _hotspots(prediction_path: Path, horizon: int = 12) -> list[dict[str, Any]]:
    predictions = _read_csv(prediction_path)
    if predictions.empty:
        return []
    score_col = f"risk_score_h{horizon}"
    level_col = f"risk_level_h{horizon}"
    comfort_col = f"comfort_h{horizon}"
    wait_col = f"pred_wait_h{horizon}"
    person_col = f"pred_person_h{horizon}"
    driver_col = f"risk_driver_h{horizon}"
    top = predictions.sort_values(score_col, ascending=False).head(12)
    return [
        {
            "datetime": row.get("datetime"),
            "node_id": row.get("node_id"),
            "risk_level": row.get(level_col),
            "risk_score": _round(row.get(score_col)),
            "comfort": _round(row.get(comfort_col), 2),
            "pred_wait": _round(row.get(wait_col), 2),
            "pred_person": _round(row.get(person_col), 2),
            "driver": row.get(driver_col),
        }
        for _, row in top.iterrows()
    ]


def build_dashboard_data(project_root: Path) -> dict[str, Any]:
    data_dir = project_root / "赛题五基于多源数据融合的文旅场景舒适度评估与风险超前预警模型与分析0805"
    graph_dir = project_root / "outputs" / "graph_v0_6_optimized"
    temporal_dir = project_root / "outputs" / "temporal_graph_v0_7_optimized"
    strategy_dir = project_root / "outputs" / "strategy_v0_7_optimized"

    return {
        "project": "BeaconNova",
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [
            _metric_block("Optimized adaptive GCN", graph_dir, "graph"),
            _metric_block("Optimized TCN-GCN", temporal_dir, "temporal_graph"),
        ],
        "strategy": _strategy_block(strategy_dir),
        "weather": _weather_block(data_dir),
        "top_edges": _edge_block(graph_dir),
        "hotspots": _hotspots(temporal_dir / "temporal_graph_predictions.csv", horizon=12),
        "assets": {
            "official_map": "assets/official_map.jpg",
            "user_map": "assets/user_map.jpg",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dashboard JSON for BeaconNova.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dashboard" / "assets" / "dashboard_data.json",
    )
    args = parser.parse_args()

    data = build_dashboard_data(args.project_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, ensure_ascii=False, indent=2)
    args.output.write_text(serialized, encoding="utf-8")
    js_output = args.output.with_suffix(".js")
    js_output.write_text("window.BEACONNOVA_DATA = " + serialized + ";`n", encoding="utf-8")
    print(f"Dashboard data written to {args.output}")
    print(f"Dashboard JS written to {js_output}")


if __name__ == "__main__":
    main()


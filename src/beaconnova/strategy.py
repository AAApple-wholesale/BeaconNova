from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LEVEL_ORDER = {"蓝色": 0, "黄色": 1, "橙色": 2, "红色": 3}
LEVEL_NAMES = ["蓝色", "黄色", "橙色", "红色"]


@dataclass(frozen=True)
class StrategyConfig:
    horizon: int = 12
    min_level: str = "黄色"
    output_dir: Path = Path("outputs") / "strategy_v0_6"


def _level_from_score(score: pd.Series) -> pd.Series:
    levels = pd.Series("蓝色", index=score.index, dtype="object")
    levels[score >= 0.35] = "黄色"
    levels[score >= 0.55] = "橙色"
    levels[score >= 0.75] = "红色"
    q75, q90, q97 = score.quantile([0.75, 0.90, 0.97]).tolist()
    levels[score >= q75] = np.maximum(levels.map(LEVEL_ORDER), LEVEL_ORDER["黄色"]).map({v: k for k, v in LEVEL_ORDER.items()})
    levels[score >= q90] = np.maximum(levels.map(LEVEL_ORDER), LEVEL_ORDER["橙色"]).map({v: k for k, v in LEVEL_ORDER.items()})
    levels[score >= q97] = np.maximum(levels.map(LEVEL_ORDER), LEVEL_ORDER["红色"]).map({v: k for k, v in LEVEL_ORDER.items()})
    return levels


def _risk_score(row: pd.Series, horizon: int, adjustments: dict[str, float]) -> float:
    comp = {
        "等待压力": float(row.get(f"risk_component_等待压力_h{horizon}", 0.0)),
        "客流压力": float(row.get(f"risk_component_客流压力_h{horizon}", 0.0)),
        "包检压力": float(row.get(f"risk_component_包检压力_h{horizon}", 0.0)),
        "增长压力": float(row.get(f"risk_component_增长压力_h{horizon}", 0.0)),
        "容量压力": float(row.get(f"risk_component_容量压力_h{horizon}", 0.0)),
        "索道压力": float(row.get(f"risk_component_索道压力_h{horizon}", 0.0)),
        "暴露风险": float(row.get(f"risk_component_暴露风险_h{horizon}", 0.0)),
        "天气压力": float(row.get(f"risk_component_天气压力_h{horizon}", 0.0)),
    }
    for key, ratio in adjustments.items():
        comp[key] = max(0.0, comp[key] * (1.0 - ratio))
    return float(
        0.28 * comp["等待压力"]
        + 0.23 * comp["客流压力"]
        + 0.11 * comp["包检压力"]
        + 0.08 * comp["增长压力"]
        + 0.14 * comp["容量压力"]
        + 0.06 * comp["索道压力"]
        + 0.04 * comp["暴露风险"]
        + 0.06 * comp["天气压力"]
    )


def _strategy_catalog(row: pd.Series, horizon: int) -> list[dict[str, object]]:
    driver = str(row.get(f"risk_driver_h{horizon}", ""))
    wait = float(row.get(f"pred_wait_h{horizon}", 0.0))
    weather = float(row.get(f"risk_component_天气压力_h{horizon}", 0.0))
    capacity = float(row.get(f"risk_component_容量压力_h{horizon}", 0.0))
    ropeway = float(row.get(f"risk_component_索道压力_h{horizon}", 0.0))
    growth = float(row.get(f"risk_component_增长压力_h{horizon}", 0.0))
    actions = []
    actions.append(
        {
            "strategy_id": "open_backup_security_channel",
            "strategy_name": "增开/预启备用安检通道",
            "trigger": "等待或包检压力升高",
            "adjustments": {"等待压力": 0.22, "客流压力": 0.10, "包检压力": 0.18},
            "wait_reduction_min": max(0.15, wait * 0.18),
            "cost_level": "中",
        }
    )
    actions.append(
        {
            "strategy_id": "dynamic_route_guidance",
            "strategy_name": "分流引导至低负载游线",
            "trigger": "容量或增长压力升高",
            "adjustments": {"客流压力": 0.16, "增长压力": 0.24, "容量压力": 0.18},
            "wait_reduction_min": max(0.10, wait * 0.12),
            "cost_level": "低",
        }
    )
    actions.append(
        {
            "strategy_id": "ropeway_pacing",
            "strategy_name": "索道入口削峰与批次放行",
            "trigger": "索道平台或轨道到达压力升高",
            "adjustments": {"索道压力": 0.28, "容量压力": 0.08, "增长压力": 0.12},
            "wait_reduction_min": max(0.08, wait * 0.10),
            "cost_level": "中",
        }
    )
    actions.append(
        {
            "strategy_id": "weather_comfort_service",
            "strategy_name": "天气保障服务点前置",
            "trigger": "高温、降雨、大风或严寒压力升高",
            "adjustments": {"天气压力": 0.35, "暴露风险": 0.20, "等待压力": 0.05},
            "wait_reduction_min": max(0.03, wait * 0.04),
            "cost_level": "低",
        }
    )
    if weather >= 0.25:
        preferred = "weather_comfort_service"
    elif ropeway >= 0.70:
        preferred = "ropeway_pacing"
    elif capacity >= 0.70 or driver == "容量压力":
        preferred = "dynamic_route_guidance"
    elif growth >= 0.60 or driver == "增长压力":
        preferred = "open_backup_security_channel"
    else:
        preferred = "dynamic_route_guidance"
    return sorted(actions, key=lambda item: item["strategy_id"] != preferred)


def simulate_strategies(predictions: pd.DataFrame, config: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizon = config.horizon
    level_col = f"risk_level_h{horizon}"
    score_col = f"risk_score_h{horizon}"
    wait_col = f"pred_wait_h{horizon}"
    min_rank = LEVEL_ORDER[config.min_level]
    records = []
    for _, row in predictions.iterrows():
        level = str(row.get(level_col, "蓝色"))
        if LEVEL_ORDER.get(level, 0) < min_rank:
            continue
        base_score = float(row.get(score_col, 0.0))
        base_comfort = float(row.get(f"comfort_h{horizon}", max(0.0, 100.0 * (1.0 - base_score))))
        base_wait = float(row.get(wait_col, 0.0))
        best = None
        for action in _strategy_catalog(row, horizon):
            adjusted_score = _risk_score(row, horizon, action["adjustments"])
            adjusted_wait = max(0.0, base_wait - float(action["wait_reduction_min"]))
            adjusted_comfort = float(np.clip(100.0 * (1.0 - adjusted_score), 0.0, 100.0))
            record = {
                "datetime": row.get("datetime"),
                "security_gate": row.get("security_gate"),
                "channel": row.get("channel"),
                "node_id": row.get("node_id"),
                "horizon_min": horizon * 5,
                "baseline_risk_level": level,
                "baseline_risk_score": base_score,
                "baseline_comfort": base_comfort,
                "baseline_wait_min": base_wait,
                "strategy_id": action["strategy_id"],
                "strategy_name": action["strategy_name"],
                "trigger": action["trigger"],
                "cost_level": action["cost_level"],
                "adjusted_risk_score": adjusted_score,
                "adjusted_comfort": adjusted_comfort,
                "adjusted_wait_min": adjusted_wait,
                "risk_score_delta": base_score - adjusted_score,
                "comfort_gain": adjusted_comfort - base_comfort,
                "wait_reduction_min": base_wait - adjusted_wait,
            }
            if best is None or record["risk_score_delta"] > best["risk_score_delta"]:
                best = record
        if best is not None:
            records.append(best)
    detail = pd.DataFrame(records)
    if detail.empty:
        return detail, pd.DataFrame()
    detail["adjusted_risk_level"] = _level_from_score(detail["adjusted_risk_score"])
    summary = (
        detail.groupby("strategy_name", as_index=False)
        .agg(
            interventions=("strategy_name", "size"),
            avg_risk_score_delta=("risk_score_delta", "mean"),
            avg_comfort_gain=("comfort_gain", "mean"),
            avg_wait_reduction_min=("wait_reduction_min", "mean"),
            baseline_orange_red=("baseline_risk_level", lambda s: int(s.isin(["橙色", "红色"]).sum())),
            adjusted_orange_red=("adjusted_risk_level", lambda s: int(s.isin(["橙色", "红色"]).sum())),
        )
        .sort_values(["avg_risk_score_delta", "avg_comfort_gain"], ascending=False)
    )
    summary["orange_red_reduction"] = summary["baseline_orange_red"] - summary["adjusted_orange_red"]
    return detail, summary


def run_strategy_simulation(prediction_path: Path, output_dir: Path, horizon: int = 12) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(prediction_path, encoding="utf-8-sig")
    detail, summary = simulate_strategies(predictions, StrategyConfig(horizon=horizon, output_dir=output_dir))
    detail_path = output_dir / "strategy_recommendations.csv"
    summary_path = output_dir / "strategy_summary.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return {"recommendations": detail_path, "summary": summary_path}

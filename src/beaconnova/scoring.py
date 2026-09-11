from __future__ import annotations

import numpy as np
import pandas as pd


def _score_wait(wait_min: pd.Series) -> pd.Series:
    return np.clip(wait_min / 20.0, 0.0, 1.0)


def _score_load(person_count: pd.Series) -> pd.Series:
    # Initial channel-level capacity proxy. It will be replaced by facility tables in later versions.
    return np.clip(person_count / 55.0, 0.0, 1.0)


def _score_growth(current: pd.Series, future: pd.Series) -> pd.Series:
    growth = (future - current) / current.replace(0, np.nan)
    return np.clip(growth.fillna(0.0), 0.0, 1.0)


def add_comfort_and_risk(predictions: pd.DataFrame, observed: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    out = predictions.merge(
        observed[["datetime", "node_id", "total_person_count", "avg_queue_wait_min", "total_bag_check_num"]],
        on=["datetime", "node_id"],
        how="left",
    )
    for horizon in horizons:
        person_col = f"pred_person_h{horizon}"
        wait_col = f"pred_wait_h{horizon}"
        if person_col not in out or wait_col not in out:
            continue
        wait_pressure = _score_wait(out[wait_col])
        load_pressure = _score_load(out[person_col])
        bag_pressure = np.clip(out["total_bag_check_num"].fillna(0.0) / 90.0, 0.0, 1.0)
        growth_pressure = _score_growth(out["total_person_count"].fillna(0.0), out[person_col])
        risk_score = 0.40 * wait_pressure + 0.35 * load_pressure + 0.15 * bag_pressure + 0.10 * growth_pressure
        comfort = np.clip(100.0 * (1.0 - risk_score), 0.0, 100.0)
        out[f"comfort_h{horizon}"] = comfort.round(2)
        out[f"risk_score_h{horizon}"] = risk_score.round(4)
        out[f"risk_level_h{horizon}"] = risk_level(risk_score)
    return out


def risk_level(risk_score: pd.Series) -> pd.Series:
    """Map risk score to a hybrid absolute/relative warning level.

    The first baseline does not yet consume facility capacity tables, so purely fixed
    thresholds can classify low-season samples as all-blue. The relative thresholds
    preserve ranking ability until node-specific business thresholds are calibrated.
    """
    q75, q90, q97 = risk_score.quantile([0.75, 0.90, 0.97]).tolist()
    levels = pd.Series("蓝色", index=risk_score.index, dtype="object")
    levels[(risk_score >= q75) | (risk_score >= 0.35)] = "黄色"
    levels[(risk_score >= q90) | (risk_score >= 0.55)] = "橙色"
    levels[(risk_score >= q97) | (risk_score >= 0.75)] = "红色"
    return levels


def risk_summary(scored: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        col = f"risk_level_h{horizon}"
        if col not in scored:
            continue
        counts = scored[col].value_counts(dropna=False).to_dict()
        for level in ["蓝色", "黄色", "橙色", "红色"]:
            rows.append({"horizon_min": horizon * 5, "risk_level": level, "count": int(counts.get(level, 0))})
    return pd.DataFrame(rows)

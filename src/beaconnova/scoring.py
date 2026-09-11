from __future__ import annotations

import numpy as np
import pandas as pd


RISK_LEVELS = ["蓝色", "黄色", "橙色", "红色"]


def _score_wait(wait_min: pd.Series) -> pd.Series:
    return np.clip(wait_min / 20.0, 0.0, 1.0)


def _score_load(person_count: pd.Series) -> pd.Series:
    return np.clip(person_count / 55.0, 0.0, 1.0)


def _score_growth(current: pd.Series, future: pd.Series) -> pd.Series:
    growth = (future - current) / current.replace(0, np.nan)
    return np.clip(growth.fillna(0.0), 0.0, 1.0)


def _max_available(out: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [col for col in columns if col in out]
    if not available:
        return pd.Series(0.0, index=out.index)
    return out[available].replace([np.inf, -np.inf], np.nan).fillna(0.0).max(axis=1)


def add_comfort_and_risk(predictions: pd.DataFrame, observed: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    base_cols = ["datetime", "node_id", "total_person_count", "avg_queue_wait_min", "total_bag_check_num"]
    optional_cols = [
        "facility_hourly_pressure",
        "facility_instant_pressure",
        "facility_bottleneck_pressure",
        "ticket_facility_hourly_pressure",
        "rail_ropeway_hourly_pressure",
        "rail_ropeway_platform_pressure",
        "ticket_ropeway_pressure",
        "facility_weather_exposure_index",
        "facility_bottleneck_risk_index",
        "ropeway_exposure_index",
        "ropeway_avg_queue_limit_min",
    ]
    merge_cols = base_cols + [col for col in optional_cols if col in observed]
    out = predictions.merge(observed[merge_cols], on=["datetime", "node_id"], how="left")
    capacity_cols = [
        "facility_hourly_pressure",
        "facility_instant_pressure",
        "facility_bottleneck_pressure",
        "ticket_facility_hourly_pressure",
    ]
    ropeway_cols = ["rail_ropeway_hourly_pressure", "rail_ropeway_platform_pressure", "ticket_ropeway_pressure"]
    for horizon in horizons:
        person_col = f"pred_person_h{horizon}"
        wait_col = f"pred_wait_h{horizon}"
        if person_col not in out or wait_col not in out:
            continue
        wait_pressure = _score_wait(out[wait_col])
        load_pressure = _score_load(out[person_col])
        bag_pressure = np.clip(out["total_bag_check_num"].fillna(0.0) / 90.0, 0.0, 1.0)
        growth_pressure = _score_growth(out["total_person_count"].fillna(0.0), out[person_col])
        capacity_pressure = np.clip(_max_available(out, capacity_cols), 0.0, 1.5) / 1.5
        ropeway_pressure = np.clip(_max_available(out, ropeway_cols), 0.0, 1.5) / 1.5
        exposure_pressure = np.clip(
            0.5 * out.get("facility_weather_exposure_index", 0.0)
            + 0.3 * out.get("facility_bottleneck_risk_index", 0.0)
            + 0.2 * out.get("ropeway_exposure_index", 0.0),
            0.0,
            1.0,
        )
        risk_score = (
            0.30 * wait_pressure
            + 0.25 * load_pressure
            + 0.12 * bag_pressure
            + 0.08 * growth_pressure
            + 0.15 * capacity_pressure
            + 0.06 * ropeway_pressure
            + 0.04 * exposure_pressure
        )
        comfort = np.clip(100.0 * (1.0 - risk_score), 0.0, 100.0)
        pressure_detail = pd.DataFrame(
            {
                "等待压力": wait_pressure,
                "客流压力": load_pressure,
                "包检压力": bag_pressure,
                "增长压力": growth_pressure,
                "容量压力": capacity_pressure,
                "索道压力": ropeway_pressure,
                "暴露风险": exposure_pressure,
            },
            index=out.index,
        )
        out[f"comfort_h{horizon}"] = comfort.round(2)
        out[f"risk_score_h{horizon}"] = risk_score.round(4)
        out[f"risk_driver_h{horizon}"] = pressure_detail.idxmax(axis=1)
        for name, values in pressure_detail.items():
            out[f"risk_component_{name}_h{horizon}"] = values.round(4)
        out[f"risk_level_h{horizon}"] = risk_level(risk_score)
    return out


def risk_level(risk_score: pd.Series) -> pd.Series:
    """Map risk score to a hybrid absolute/relative warning level."""
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
        driver_col = f"risk_driver_h{horizon}"
        if col not in scored:
            continue
        counts = scored[col].value_counts(dropna=False).to_dict()
        for level in RISK_LEVELS:
            rows.append({"horizon_min": horizon * 5, "risk_level": level, "count": int(counts.get(level, 0))})
        if driver_col in scored:
            for driver, count in scored[driver_col].value_counts(dropna=False).to_dict().items():
                rows.append({"horizon_min": horizon * 5, "risk_level": f"主导因素:{driver}", "count": int(count)})
    return pd.DataFrame(rows)

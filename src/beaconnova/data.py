from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import FACILITY_FILE, RAIL_FILE, ROPEWAY_FILE, SECURITY_FILE, TICKET_FILE


def _parse_datetime(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        date_series.astype(str).str.strip() + " " + time_series.astype(str).str.strip(),
        errors="coerce",
    )


def _to_number(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


def _read_first_nonempty_excel(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frames = []
    excel = pd.ExcelFile(path)
    for sheet in excel.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _contains(text: pd.Series, pattern: str) -> pd.Series:
    return text.astype(str).str.contains(pattern, na=False, regex=False).astype(float)


def load_security_data(data_dir: Path) -> pd.DataFrame:
    """Load 5-minute security-gate observations."""
    path = data_dir / SECURITY_FILE
    df = pd.read_csv(path, encoding="gbk")
    df["datetime"] = _parse_datetime(df["date"], df["time_window"])
    df = df.dropna(subset=["datetime"]).copy()
    df["total_person_count"] = pd.to_numeric(df["total_person_count"], errors="coerce").fillna(0.0)
    df["avg_queue_wait_min"] = pd.to_numeric(df["avg_queue_wait_min"], errors="coerce").fillna(0.0)
    df["total_bag_check_num"] = pd.to_numeric(df["total_bag_check_num"], errors="coerce").fillna(0.0)
    df["node_id"] = df["security_gate"].astype(str) + "/" + df["channel"].astype(str)
    df = df.sort_values(["node_id", "datetime"]).reset_index(drop=True)
    return df


def load_rail_data(data_dir: Path) -> pd.DataFrame:
    """Load rail arrivals and aggregate repeated trains in the same 5-minute window."""
    path = data_dir / RAIL_FILE
    df = pd.read_csv(path, encoding="gbk")
    df["datetime"] = _parse_datetime(df["date"], df["train_time"])
    df = df.dropna(subset=["datetime"]).copy()
    df["passenger_count"] = pd.to_numeric(df["passenger_count"], errors="coerce").fillna(0.0)
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce").fillna(0.0)
    df["datetime"] = df["datetime"].dt.floor("5min")
    agg = (
        df.groupby("datetime", as_index=False)
        .agg(
            rail_passengers=("passenger_count", "sum"),
            rail_capacity=("capacity", "sum"),
        )
        .sort_values("datetime")
    )
    agg["rail_load_ratio"] = agg["rail_passengers"] / agg["rail_capacity"].replace(0, pd.NA)
    agg["rail_load_ratio"] = agg["rail_load_ratio"].fillna(0.0)
    return agg


def load_ticket_aggregates(data_dir: Path, chunksize: int = 500_000) -> pd.DataFrame:
    """Aggregate the large ticket stream to 5-minute features without loading it all at once."""
    path = data_dir / TICKET_FILE
    parts: list[pd.DataFrame] = []
    usecols = ["date", "time", "ticket_type", "channel", "season_tag"]
    for chunk in pd.read_csv(path, encoding="utf-8-sig", usecols=usecols, chunksize=chunksize):
        chunk["datetime"] = _parse_datetime(chunk["date"], chunk["time"]).dt.floor("5min")
        chunk = chunk.dropna(subset=["datetime"])
        if chunk.empty:
            continue
        base = chunk.groupby("datetime").size().rename("ticket_count").to_frame()
        type_counts = pd.crosstab(chunk["datetime"], chunk["ticket_type"].astype(str)).add_prefix("ticket_type_")
        channel_counts = pd.crosstab(chunk["datetime"], chunk["channel"].astype(str)).add_prefix("ticket_channel_")
        season_counts = pd.crosstab(chunk["datetime"], chunk["season_tag"].astype(str)).add_prefix("season_")
        part = base.join(type_counts, how="left").join(channel_counts, how="left").join(season_counts, how="left")
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["datetime", "ticket_count"])
    agg = pd.concat(parts).groupby(level=0).sum().sort_index()
    agg.index.name = "datetime"
    agg = agg.reset_index()
    count = agg["ticket_count"].replace(0, pd.NA)
    for col in list(agg.columns):
        if col.startswith("ticket_type_") or col.startswith("ticket_channel_") or col.startswith("season_"):
            agg[col + "_share"] = (agg[col] / count).fillna(0.0)
    return agg.fillna(0.0)


def load_facility_capacity(data_dir: Path) -> pd.DataFrame:
    """Load scenic-spot capacity and service-efficiency parameters."""
    df = _read_first_nonempty_excel(data_dir / FACILITY_FILE)
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["point_name"] = df["点位名称"].astype(str).str.strip()
    out["effective_area_sqm"] = _to_number(df["有效面积 (㎡)"])
    out["instant_capacity"] = _to_number(df["瞬时最大容纳量 (人)"])
    out["hourly_capacity"] = _to_number(df["小时极限通行量 (人次)"])
    out["max_dwell_min"] = _to_number(df["合理停留最大时长 (min)"])
    out["facility_text"] = df["配套设施配置"].astype(str).str.strip()
    out["risk_text"] = df["核心风险特征"].astype(str).str.strip()
    out["has_rain_shelter"] = _contains(out["facility_text"], "防雨棚")
    out["has_cooling"] = _contains(out["facility_text"], "喷淋")
    out["has_shade"] = _contains(out["facility_text"], "遮阳")
    out["is_exposed"] = _contains(out["facility_text"], "无遮挡")
    risk_text = out["risk_text"]
    out["steep_slope_risk"] = risk_text.astype(str).str.contains("陡坡|坡", na=False, regex=True).astype(float)
    out["confluence_risk"] = risk_text.astype(str).str.contains("汇流|对冲|集中", na=False, regex=True).astype(float)
    out["narrow_path_risk"] = risk_text.astype(str).str.contains("窄|狭小", na=False, regex=True).astype(float)
    out["weather_exposure_index"] = np.clip(1.0 - 0.35 * out["has_rain_shelter"] - 0.35 * out["has_cooling"] - 0.20 * out["has_shade"] + 0.25 * out["is_exposed"], 0.0, 1.0)
    out["bottleneck_risk_index"] = np.clip(0.45 * out["steep_slope_risk"] + 0.35 * out["confluence_risk"] + 0.20 * out["narrow_path_risk"], 0.0, 1.0)
    out["service_rate_per_min"] = out["hourly_capacity"] / 60.0
    total_capacity = out["instant_capacity"].sum()
    out["instant_capacity_share"] = out["instant_capacity"] / total_capacity if total_capacity else 0.0
    return out.fillna(0.0)


def load_ropeway_capacity(data_dir: Path) -> pd.DataFrame:
    """Load north/south ropeway operation parameters."""
    df = _read_first_nonempty_excel(data_dir / ROPEWAY_FILE)
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["ropeway_name"] = df["索道名称"].astype(str).str.strip()
    out["line"] = np.where(out["ropeway_name"].str.contains("北", na=False), "north", "south")
    out["vehicle_capacity"] = _to_number(df["单车厢额定载客 (人)"])
    out["min_departure_interval_sec"] = _to_number(df["最小发车间隔 (秒)"])
    out["theoretical_hourly_capacity"] = _to_number(df["理论单向小时最大运力 (人次)"])
    out["saturated_hourly_capacity"] = _to_number(df["实际运营饱和运力 (人次 / 小时)"])
    out["platform_instant_capacity"] = _to_number(df["入口平台瞬时最大容纳量 (人)"])
    out["queue_limit_min"] = _to_number(df["合理排队时长上限 (min)"])
    out["facility_text"] = df["配套设施配置"].astype(str).str.strip()
    out["risk_text"] = df["核心运营风险特征"].astype(str).str.strip()
    out["departures_per_hour"] = 3600.0 / out["min_departure_interval_sec"].replace(0, pd.NA)
    out["operating_derate_ratio"] = out["saturated_hourly_capacity"] / out["theoretical_hourly_capacity"].replace(0, pd.NA)
    out["has_cooling"] = _contains(out["facility_text"], "喷淋")
    out["has_shade"] = _contains(out["facility_text"], "遮阳")
    out["ropeway_exposure_index"] = np.clip(1.0 - 0.45 * out["has_cooling"] - 0.30 * out["has_shade"], 0.0, 1.0)
    return out.fillna(0.0)

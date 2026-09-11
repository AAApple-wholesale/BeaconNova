from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import RAIL_FILE, SECURITY_FILE, TICKET_FILE


def _parse_datetime(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        date_series.astype(str).str.strip() + " " + time_series.astype(str).str.strip(),
        errors="coerce",
    )


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

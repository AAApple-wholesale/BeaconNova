from __future__ import annotations

import numpy as np
import pandas as pd


LAG_WINDOWS = (1, 2, 3, 6, 12, 18, 24, 36)
ROLLING_WINDOWS = (3, 6, 12, 18, 24, 36)
PEAK_WINDOWS = (6, 12, 24, 36)
PROPAGATION_LAGS = (1, 3, 6, 12)
NATIONAL_DAY_2025 = pd.date_range("2025-10-01", "2025-10-08", freq="D")
MAKEUP_WORKDAYS_2025 = pd.to_datetime(["2025-09-28", "2025-10-11"])


def _safe_divide(numerator: pd.Series | float, denominator: float) -> pd.Series | float:
    if denominator <= 0:
        if isinstance(numerator, pd.Series):
            return pd.Series(0.0, index=numerator.index)
        return 0.0
    return numerator / denominator


def _summary_value(df: pd.DataFrame | None, column: str, default: float = 0.0, agg: str = "sum") -> float:
    if df is None or df.empty or column not in df:
        return default
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return default
    if agg == "min":
        return float(values.min())
    if agg == "max":
        return float(values.max())
    if agg == "mean":
        return float(values.mean())
    return float(values.sum())


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out["datetime"]
    out["hour"] = dt.dt.hour
    out["minute"] = dt.dt.minute
    out["dayofweek"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["dayofyear"] = dt.dt.dayofyear
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["slot_5min"] = out["hour"] * 12 + (out["minute"] // 5)
    minute_of_day = out["hour"] * 60 + out["minute"]
    out["minute_of_day"] = minute_of_day
    out["sin_slot"] = np.sin(2 * np.pi * out["slot_5min"] / 288)
    out["cos_slot"] = np.cos(2 * np.pi * out["slot_5min"] / 288)
    out["sin_week"] = np.sin(2 * np.pi * (out["dayofweek"] * 288 + out["slot_5min"]) / (7 * 288))
    out["cos_week"] = np.cos(2 * np.pi * (out["dayofweek"] * 288 + out["slot_5min"]) / (7 * 288))
    out["is_opening_ramp"] = ((minute_of_day >= 420) & (minute_of_day < 540)).astype(int)
    out["is_morning_peak"] = ((minute_of_day >= 540) & (minute_of_day < 690)).astype(int)
    out["is_midday_peak"] = ((minute_of_day >= 690) & (minute_of_day < 840)).astype(int)
    out["is_afternoon_peak"] = ((minute_of_day >= 840) & (minute_of_day < 990)).astype(int)
    out["is_closing_period"] = ((minute_of_day >= 990) & (minute_of_day < 1110)).astype(int)
    out["is_business_hour"] = ((minute_of_day >= 420) & (minute_of_day < 1110)).astype(int)
    out["peak_period_code"] = (
        out["is_opening_ramp"]
        + 2 * out["is_morning_peak"]
        + 3 * out["is_midday_peak"]
        + 4 * out["is_afternoon_peak"]
        + 5 * out["is_closing_period"]
    )
    out = add_calendar_features(out)
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    date = out["datetime"].dt.normalize()
    holidays = pd.Series(NATIONAL_DAY_2025)
    out["is_holiday"] = date.isin(holidays).astype(int)
    out["is_makeup_workday"] = date.isin(pd.Series(MAKEUP_WORKDAYS_2025)).astype(int)
    out["is_effective_weekend"] = ((out["is_weekend"] == 1) & (out["is_makeup_workday"] == 0) | (out["is_holiday"] == 1)).astype(int)
    out["is_saturday"] = (out["dayofweek"] == 5).astype(int)
    out["is_sunday"] = (out["dayofweek"] == 6).astype(int)
    out["is_friday"] = (out["dayofweek"] == 4).astype(int)
    out["is_monday"] = (out["dayofweek"] == 0).astype(int)
    out["is_summer_vacation"] = ((date >= pd.Timestamp("2025-07-01")) & (date <= pd.Timestamp("2025-08-31"))).astype(int)
    out["holiday_seq"] = 0
    for idx, holiday in enumerate(NATIONAL_DAY_2025, start=1):
        out.loc[date == holiday, "holiday_seq"] = idx
    holiday_values = holidays.astype("int64").to_numpy()
    date_values = date.astype("int64").to_numpy()
    days_to = []
    days_since = []
    one_day = 24 * 60 * 60 * 1_000_000_000
    for value in date_values:
        future = holiday_values[holiday_values >= value]
        past = holiday_values[holiday_values <= value]
        days_to.append(int((future.min() - value) / one_day) if len(future) else 999)
        days_since.append(int((value - past.max()) / one_day) if len(past) else 999)
    out["days_to_holiday"] = days_to
    out["days_since_holiday"] = days_since
    out["is_pre_holiday_1d"] = (out["days_to_holiday"] == 1).astype(int)
    out["is_pre_holiday_3d"] = out["days_to_holiday"].between(1, 3).astype(int)
    out["is_post_holiday_1d"] = (out["days_since_holiday"] == 1).astype(int)
    out["is_post_holiday_3d"] = out["days_since_holiday"].between(1, 3).astype(int)
    out["is_holiday_adjacent"] = ((out["days_to_holiday"] <= 3) | (out["days_since_holiday"] <= 3)).astype(int)
    out["holiday_phase_sin"] = np.sin(2 * np.pi * out["holiday_seq"] / max(len(NATIONAL_DAY_2025), 1)) * out["is_holiday"]
    out["holiday_phase_cos"] = np.cos(2 * np.pi * out["holiday_seq"] / max(len(NATIONAL_DAY_2025), 1)) * out["is_holiday"]
    out["days_to_weekend"] = (5 - out["dayofweek"]).clip(lower=0)
    out["weekend_tail"] = ((out["is_sunday"] == 1) | (out["is_monday"] == 1)).astype(int)
    out["weekend_ramp"] = ((out["is_friday"] == 1) | (out["is_saturday"] == 1)).astype(int)
    out["calendar_demand_weight"] = (
        1.0
        + 0.18 * out["is_effective_weekend"]
        + 0.30 * out["is_holiday"]
        + 0.10 * out["is_pre_holiday_3d"]
        + 0.08 * out["is_summer_vacation"]
    )
    return out


def add_security_history_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["node_id", "datetime"]).copy()
    group = out.groupby("node_id", group_keys=False)
    base_cols = ["total_person_count", "avg_queue_wait_min", "total_bag_check_num"]

    new_cols: dict[str, pd.Series] = {}
    for col in base_cols:
        shifted = group[col].shift(1)
        for lag in LAG_WINDOWS:
            new_cols[f"{col}_lag_{lag}"] = group[col].shift(lag)
        for window in ROLLING_WINDOWS:
            rolling = shifted.groupby(out["node_id"]).rolling(window)
            roll_mean = rolling.mean().reset_index(level=0, drop=True)
            roll_max = rolling.max().reset_index(level=0, drop=True)
            roll_min = rolling.min().reset_index(level=0, drop=True)
            roll_std = rolling.std().reset_index(level=0, drop=True)
            new_cols[f"{col}_roll_mean_{window}"] = roll_mean
            new_cols[f"{col}_roll_max_{window}"] = roll_max
            new_cols[f"{col}_roll_min_{window}"] = roll_min
            new_cols[f"{col}_roll_std_{window}"] = roll_std
            if window in PEAK_WINDOWS:
                new_cols[f"{col}_roll_q90_{window}"] = rolling.quantile(0.90).reset_index(level=0, drop=True)
                new_cols[f"{col}_peak_gap_{window}"] = roll_max - out[col]
                new_cols[f"{col}_peak_ratio_{window}"] = out[col] / roll_max.replace(0, np.nan)
                new_cols[f"{col}_above_q90_{window}"] = (out[col] >= new_cols[f"{col}_roll_q90_{window}"]).astype(float)

    new_cols["person_diff_1"] = group["total_person_count"].diff(1)
    new_cols["person_diff_3"] = group["total_person_count"].diff(3)
    new_cols["person_diff_6"] = group["total_person_count"].diff(6)
    new_cols["wait_diff_1"] = group["avg_queue_wait_min"].diff(1)
    new_cols["wait_diff_3"] = group["avg_queue_wait_min"].diff(3)
    new_cols["wait_diff_6"] = group["avg_queue_wait_min"].diff(6)
    new_cols["bag_diff_1"] = group["total_bag_check_num"].diff(1)
    new_cols["person_momentum_30min"] = group["total_person_count"].shift(1) - group["total_person_count"].shift(6)
    new_cols["person_momentum_60min"] = group["total_person_count"].shift(1) - group["total_person_count"].shift(12)
    new_cols["wait_momentum_30min"] = group["avg_queue_wait_min"].shift(1) - group["avg_queue_wait_min"].shift(6)
    new_cols["wait_momentum_60min"] = group["avg_queue_wait_min"].shift(1) - group["avg_queue_wait_min"].shift(12)
    new_cols["wait_accel_15min"] = group["avg_queue_wait_min"].diff(1) - group["avg_queue_wait_min"].diff(3)
    new_cols["wait_accel_30min"] = group["avg_queue_wait_min"].diff(3) - group["avg_queue_wait_min"].diff(6)
    new_cols["bag_per_person"] = out["total_bag_check_num"] / out["total_person_count"].replace(0, np.nan)
    new_cols["wait_per_person"] = out["avg_queue_wait_min"] / out["total_person_count"].replace(0, np.nan)
    new_cols["wait_level_x_momentum_30min"] = out["avg_queue_wait_min"] * new_cols["wait_momentum_30min"]

    add = pd.DataFrame(new_cols, index=out.index).replace([np.inf, -np.inf], np.nan)
    return pd.concat([out, add], axis=1).copy()


def add_gate_context_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    gate = (
        out.groupby(["datetime", "security_gate"], as_index=False)
        .agg(
            gate_person_count=("total_person_count", "sum"),
            gate_wait_mean=("avg_queue_wait_min", "mean"),
            gate_wait_max=("avg_queue_wait_min", "max"),
            gate_bag_count=("total_bag_check_num", "sum"),
            gate_channel_count=("channel", "nunique"),
        )
    )
    out = out.merge(gate, on=["datetime", "security_gate"], how="left")
    system = (
        out.groupby("datetime", as_index=False)
        .agg(
            system_person_count=("total_person_count", "sum"),
            system_wait_mean=("avg_queue_wait_min", "mean"),
            system_wait_max=("avg_queue_wait_min", "max"),
            system_bag_count=("total_bag_check_num", "sum"),
        )
    )
    out = out.merge(system, on="datetime", how="left")
    out["channel_load_share"] = out["total_person_count"] / out["gate_person_count"].replace(0, np.nan)
    out["gate_system_share"] = out["gate_person_count"] / out["system_person_count"].replace(0, np.nan)
    out["channel_wait_gap"] = out["avg_queue_wait_min"] - out["gate_wait_mean"]
    out["gate_wait_gap"] = out["gate_wait_mean"] - out["system_wait_mean"]
    out["gate_flow_intensity"] = out["gate_person_count"] / out["gate_bag_count"].replace(0, np.nan)
    cols = ["channel_load_share", "gate_system_share", "gate_flow_intensity", "channel_wait_gap", "gate_wait_gap"]
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def add_congestion_propagation_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["node_id", "datetime"]).copy()
    group = out.groupby("node_id", group_keys=False)
    new_cols: dict[str, pd.Series] = {}
    context_cols = [
        "gate_person_count",
        "gate_wait_mean",
        "gate_wait_max",
        "system_person_count",
        "system_wait_mean",
        "system_wait_max",
    ]
    for col in context_cols:
        for lag in PROPAGATION_LAGS:
            new_cols[f"{col}_lag_{lag}"] = group[col].shift(lag)
        shifted = group[col].shift(1)
        new_cols[f"{col}_roll_mean_6"] = shifted.groupby(out["node_id"]).rolling(6).mean().reset_index(level=0, drop=True)
        new_cols[f"{col}_roll_max_12"] = shifted.groupby(out["node_id"]).rolling(12).max().reset_index(level=0, drop=True)

    for lag in PROPAGATION_LAGS:
        gate_person = new_cols[f"gate_person_count_lag_{lag}"]
        gate_wait = new_cols[f"gate_wait_mean_lag_{lag}"]
        system_person = new_cols[f"system_person_count_lag_{lag}"]
        system_wait = new_cols[f"system_wait_mean_lag_{lag}"]
        node_person = out.get(f"total_person_count_lag_{lag}", pd.Series(0.0, index=out.index))
        node_wait = out.get(f"avg_queue_wait_min_lag_{lag}", pd.Series(0.0, index=out.index))
        new_cols[f"upstream_gate_person_lag_{lag}"] = (gate_person - node_person).clip(lower=0.0)
        new_cols[f"upstream_gate_wait_lag_{lag}"] = gate_wait
        new_cols[f"upstream_system_person_lag_{lag}"] = (system_person - gate_person).clip(lower=0.0)
        new_cols[f"upstream_system_wait_lag_{lag}"] = system_wait
        new_cols[f"propagated_person_pressure_lag_{lag}"] = (0.55 * gate_person + 0.45 * system_person) / 1000.0
        new_cols[f"propagated_wait_pressure_lag_{lag}"] = 0.60 * gate_wait + 0.40 * system_wait
        new_cols[f"relative_wait_pressure_lag_{lag}"] = node_wait - system_wait

    add = pd.DataFrame(new_cols, index=out.index).replace([np.inf, -np.inf], np.nan)
    return pd.concat([out, add], axis=1).copy()


def add_rail_features(security_df: pd.DataFrame, rail_df: pd.DataFrame) -> pd.DataFrame:
    out = security_df.copy()
    timeline = pd.DataFrame({"datetime": sorted(out["datetime"].dropna().unique())})
    rail = timeline.merge(rail_df, on="datetime", how="left").fillna(
        {"rail_passengers": 0.0, "rail_capacity": 0.0, "rail_load_ratio": 0.0}
    )
    rail = rail.sort_values("datetime")
    rail["rail_passengers_30min"] = rail["rail_passengers"].rolling(6, min_periods=1).sum()
    rail["rail_passengers_60min"] = rail["rail_passengers"].rolling(12, min_periods=1).sum()
    rail["rail_passengers_120min"] = rail["rail_passengers"].rolling(24, min_periods=1).sum()
    rail["rail_passengers_diff_30min"] = rail["rail_passengers_30min"] - rail["rail_passengers_30min"].shift(6)
    rail["rail_load_ratio_60min"] = rail["rail_load_ratio"].rolling(12, min_periods=1).mean()
    return out.merge(rail.fillna(0.0), on="datetime", how="left")


def add_ticket_features(security_df: pd.DataFrame, ticket_df: pd.DataFrame | None) -> pd.DataFrame:
    out = security_df.copy()
    if ticket_df is None or ticket_df.empty:
        out["ticket_count"] = 0.0
        out["ticket_count_30min"] = 0.0
        out["ticket_count_60min"] = 0.0
        out["ticket_count_120min"] = 0.0
        out["ticket_count_day_cum"] = 0.0
        out["ticket_count_diff_30min"] = 0.0
        return out
    timeline = pd.DataFrame({"datetime": sorted(out["datetime"].dropna().unique())})
    ticket = timeline.merge(ticket_df, on="datetime", how="left").fillna(0.0).sort_values("datetime")
    ticket["ticket_count_30min"] = ticket["ticket_count"].rolling(6, min_periods=1).sum()
    ticket["ticket_count_60min"] = ticket["ticket_count"].rolling(12, min_periods=1).sum()
    ticket["ticket_count_120min"] = ticket["ticket_count"].rolling(24, min_periods=1).sum()
    ticket["ticket_count_diff_30min"] = ticket["ticket_count_30min"] - ticket["ticket_count_30min"].shift(6)
    ticket["ticket_count_day_cum"] = ticket.groupby(ticket["datetime"].dt.normalize())["ticket_count"].cumsum()
    share_cols = [c for c in ticket.columns if c.endswith("_share")]
    keep_cols = [
        "datetime",
        "ticket_count",
        "ticket_count_30min",
        "ticket_count_60min",
        "ticket_count_120min",
        "ticket_count_day_cum",
        "ticket_count_diff_30min",
    ] + share_cols
    return out.merge(ticket[keep_cols], on="datetime", how="left").fillna(0.0)


def add_facility_features(
    security_df: pd.DataFrame,
    facility_df: pd.DataFrame | None = None,
    ropeway_df: pd.DataFrame | None = None,
    weather_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Project static carrying-capacity tables into dynamic pressure features."""
    out = security_df.copy()
    facility_instant = _summary_value(facility_df, "instant_capacity")
    facility_hourly = _summary_value(facility_df, "hourly_capacity")
    facility_bottleneck = _summary_value(facility_df, "instant_capacity", agg="min")
    facility_exposure = _summary_value(facility_df, "weather_exposure_index", agg="mean")
    facility_bottleneck_risk = _summary_value(facility_df, "bottleneck_risk_index", agg="mean")
    facility_dwell = _summary_value(facility_df, "max_dwell_min", agg="mean")
    ropeway_hourly = _summary_value(ropeway_df, "saturated_hourly_capacity")
    ropeway_platform = _summary_value(ropeway_df, "platform_instant_capacity")
    ropeway_queue_limit = _summary_value(ropeway_df, "queue_limit_min", agg="mean")
    ropeway_derate = _summary_value(ropeway_df, "operating_derate_ratio", agg="mean")
    ropeway_exposure = _summary_value(ropeway_df, "ropeway_exposure_index", agg="mean")

    out["facility_total_instant_capacity"] = facility_instant
    out["facility_total_hourly_capacity"] = facility_hourly
    out["facility_min_instant_capacity"] = facility_bottleneck
    out["facility_weather_exposure_index"] = facility_exposure
    out["facility_bottleneck_risk_index"] = facility_bottleneck_risk
    out["facility_avg_dwell_min"] = facility_dwell
    out["ropeway_total_hourly_capacity"] = ropeway_hourly
    out["ropeway_total_platform_capacity"] = ropeway_platform
    out["ropeway_avg_queue_limit_min"] = ropeway_queue_limit
    out["ropeway_operating_derate_ratio"] = ropeway_derate
    out["ropeway_exposure_index"] = ropeway_exposure

    out["security_hourly_flow_est"] = out["system_person_count"].fillna(0.0) * 12.0
    out["facility_hourly_pressure"] = _safe_divide(out["security_hourly_flow_est"], facility_hourly)
    out["facility_instant_pressure"] = _safe_divide(out["system_person_count"].fillna(0.0), facility_instant)
    out["facility_bottleneck_pressure"] = _safe_divide(out["system_person_count"].fillna(0.0), facility_bottleneck)
    out["ticket_facility_hourly_pressure"] = _safe_divide(out.get("ticket_count_60min", pd.Series(0.0, index=out.index)), facility_hourly)
    out["ticket_facility_120min_pressure"] = _safe_divide(out.get("ticket_count_120min", pd.Series(0.0, index=out.index)), facility_hourly * 2.0)
    out["rail_ropeway_hourly_pressure"] = _safe_divide(out.get("rail_passengers_60min", pd.Series(0.0, index=out.index)), ropeway_hourly)
    out["rail_ropeway_120min_pressure"] = _safe_divide(out.get("rail_passengers_120min", pd.Series(0.0, index=out.index)), ropeway_hourly * 2.0)
    out["rail_ropeway_platform_pressure"] = _safe_divide(out.get("rail_passengers_30min", pd.Series(0.0, index=out.index)), ropeway_platform)
    out["ticket_ropeway_pressure"] = _safe_divide(out.get("ticket_count_60min", pd.Series(0.0, index=out.index)), ropeway_hourly)

    pressure_cols = [
        "facility_hourly_pressure",
        "facility_instant_pressure",
        "facility_bottleneck_pressure",
        "ticket_facility_hourly_pressure",
        "ticket_facility_120min_pressure",
        "rail_ropeway_hourly_pressure",
        "rail_ropeway_120min_pressure",
        "rail_ropeway_platform_pressure",
        "ticket_ropeway_pressure",
    ]
    out[pressure_cols] = out[pressure_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for col in pressure_cols:
        out[col] = out[col].clip(lower=0.0)
    return out




def add_weather_features(security_df: pd.DataFrame, weather_df: pd.DataFrame | None) -> pd.DataFrame:
    out = security_df.copy()
    weather_cols = [
        "weather_temp_c",
        "weather_apparent_temp_c",
        "weather_humidity",
        "weather_rain_mm",
        "weather_precipitation_mm",
        "weather_rain_flag",
        "weather_wind_speed_mps",
        "weather_wind_gust_mps",
        "weather_code",
        "weather_cloud_cover",
        "weather_heat_stress",
        "weather_cold_stress",
        "weather_wind_stress",
        "weather_comfort_penalty",
        "weather_is_proxy",
    ]
    derived_cols = [
        "weather_rain_1h",
        "weather_rain_3h",
        "weather_rain_6h",
        "weather_temp_diff_1h",
        "weather_temp_diff_3h",
        "weather_temp_diff_6h",
        "weather_apparent_temp_diff_1h",
        "weather_wind_diff_1h",
        "weather_wind_diff_3h",
        "weather_wind_diff_6h",
        "weather_penalty_diff_1h",
        "weather_penalty_diff_3h",
        "weather_rain_start_1h",
        "weather_rain_intensify_1h",
        "weather_abrupt_change_score",
    ]
    if weather_df is None or weather_df.empty:
        for col in weather_cols + derived_cols:
            out[col] = 0.0
        out["weather_is_proxy"] = 1.0
        return out
    weather = weather_df.copy()
    weather["datetime"] = pd.to_datetime(weather["datetime"], errors="coerce")
    weather = weather.dropna(subset=["datetime"]).sort_values("datetime")
    timeline = pd.DataFrame({"datetime": sorted(out["datetime"].dropna().unique())})
    merged = pd.merge_asof(
        timeline.sort_values("datetime"),
        weather.sort_values("datetime"),
        on="datetime",
        direction="backward",
        tolerance=pd.Timedelta("90min"),
    )
    merged = merged.sort_values("datetime")
    for col in weather_cols:
        if col not in merged:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged[weather_cols] = merged[weather_cols].ffill().bfill().fillna(0.0)
    # Security samples are 5-minute slots: 12/36/72 slots represent 1/3/6 hours.
    merged["weather_rain_1h"] = merged["weather_precipitation_mm"].rolling(12, min_periods=1).sum()
    merged["weather_rain_3h"] = merged["weather_precipitation_mm"].rolling(36, min_periods=1).sum()
    merged["weather_rain_6h"] = merged["weather_precipitation_mm"].rolling(72, min_periods=1).sum()
    merged["weather_temp_diff_1h"] = merged["weather_temp_c"] - merged["weather_temp_c"].shift(12)
    merged["weather_temp_diff_3h"] = merged["weather_temp_c"] - merged["weather_temp_c"].shift(36)
    merged["weather_temp_diff_6h"] = merged["weather_temp_c"] - merged["weather_temp_c"].shift(72)
    merged["weather_apparent_temp_diff_1h"] = merged["weather_apparent_temp_c"] - merged["weather_apparent_temp_c"].shift(12)
    merged["weather_wind_diff_1h"] = merged["weather_wind_speed_mps"] - merged["weather_wind_speed_mps"].shift(12)
    merged["weather_wind_diff_3h"] = merged["weather_wind_speed_mps"] - merged["weather_wind_speed_mps"].shift(36)
    merged["weather_wind_diff_6h"] = merged["weather_wind_speed_mps"] - merged["weather_wind_speed_mps"].shift(72)
    merged["weather_penalty_diff_1h"] = merged["weather_comfort_penalty"] - merged["weather_comfort_penalty"].shift(12)
    merged["weather_penalty_diff_3h"] = merged["weather_comfort_penalty"] - merged["weather_comfort_penalty"].shift(36)
    rain_lag_1h = merged["weather_rain_flag"].shift(12).fillna(0.0)
    rain_1h_lag = merged["weather_rain_1h"].shift(12).fillna(0.0)
    merged["weather_rain_start_1h"] = ((merged["weather_rain_flag"] > 0) & (rain_lag_1h <= 0)).astype(float)
    merged["weather_rain_intensify_1h"] = (merged["weather_rain_1h"] - rain_1h_lag).clip(lower=0.0)
    merged["weather_abrupt_change_score"] = np.clip(
        0.22 * merged["weather_temp_diff_1h"].abs() / 6.0
        + 0.20 * merged["weather_wind_diff_1h"].abs() / 5.0
        + 0.28 * merged["weather_rain_intensify_1h"] / 5.0
        + 0.30 * merged["weather_penalty_diff_1h"].clip(lower=0.0),
        0.0,
        1.0,
    )
    merged[derived_cols] = merged[derived_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.merge(merged[["datetime"] + weather_cols + derived_cols], on="datetime", how="left").fillna(0.0)


def add_targets(df: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    out = df.sort_values(["node_id", "datetime"]).copy()
    group = out.groupby("node_id", group_keys=False)
    for horizon in horizons:
        out[f"target_person_h{horizon}"] = group["total_person_count"].shift(-horizon)
        out[f"target_wait_h{horizon}"] = group["avg_queue_wait_min"].shift(-horizon)
    return out


def build_feature_frame(
    security_df: pd.DataFrame,
    rail_df: pd.DataFrame,
    horizons: tuple[int, ...],
    ticket_df: pd.DataFrame | None = None,
    facility_df: pd.DataFrame | None = None,
    ropeway_df: pd.DataFrame | None = None,
    weather_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = add_time_features(security_df)
    df = add_gate_context_features(df)
    df = add_security_history_features(df)
    df = add_congestion_propagation_features(df)
    df = add_rail_features(df, rail_df)
    df = add_ticket_features(df, ticket_df)
    df = add_facility_features(df, facility_df, ropeway_df)
    df = add_weather_features(df, weather_df)
    df = add_targets(df, horizons)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "date",
        "time_window",
        "security_gate",
        "channel",
        "datetime",
        "node_id",
    }
    target_prefixes = ("target_",)
    cols = []
    for col in df.columns:
        if col in excluded or col.startswith(target_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols



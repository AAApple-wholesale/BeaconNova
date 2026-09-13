from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import json

import numpy as np
import pandas as pd


BADALING_LATITUDE = 40.3597
BADALING_LONGITUDE = 116.0201
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
]


@dataclass(frozen=True)
class WeatherFetchConfig:
    latitude: float = BADALING_LATITUDE
    longitude: float = BADALING_LONGITUDE
    timezone: str = "Asia/Shanghai"
    start_date: str = "2025-07-01"
    end_date: str = "2025-12-31"
    output_path: Path | None = None


def open_meteo_archive_url(config: WeatherFetchConfig) -> str:
    params = {
        "latitude": config.latitude,
        "longitude": config.longitude,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": config.timezone,
        "wind_speed_unit": "ms",
    }
    return OPEN_METEO_ARCHIVE_URL + "?" + urlencode(params)


def _column(hourly: dict, name: str, default: float = 0.0) -> pd.Series:
    values = hourly.get(name)
    if values is None:
        values = [default] * len(hourly.get("time", []))
    return pd.to_numeric(pd.Series(values), errors="coerce").fillna(default)


def normalize_open_meteo_payload(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly") or {}
    if not hourly.get("time"):
        raise ValueError("Open-Meteo response does not contain hourly.time")
    out = pd.DataFrame({"datetime": pd.to_datetime(hourly["time"], errors="coerce")})
    out = out.dropna(subset=["datetime"]).copy()
    out["weather_temp_c"] = _column(hourly, "temperature_2m")
    out["weather_humidity"] = _column(hourly, "relative_humidity_2m") / 100.0
    out["weather_apparent_temp_c"] = _column(hourly, "apparent_temperature")
    out["weather_rain_mm"] = _column(hourly, "rain")
    precipitation = _column(hourly, "precipitation")
    out["weather_precipitation_mm"] = precipitation
    out["weather_rain_mm"] = np.maximum(out["weather_rain_mm"], precipitation)
    out["weather_rain_flag"] = (out["weather_rain_mm"] > 0).astype(float)
    out["weather_wind_speed_mps"] = _column(hourly, "wind_speed_10m")
    out["weather_wind_gust_mps"] = _column(hourly, "wind_gusts_10m")
    out["weather_code"] = _column(hourly, "weather_code")
    out["weather_cloud_cover"] = _column(hourly, "cloud_cover") / 100.0
    out["weather_heat_stress"] = np.clip((out["weather_apparent_temp_c"] - 28.0) / 10.0, 0.0, 1.0)
    out["weather_cold_stress"] = np.clip((5.0 - out["weather_apparent_temp_c"]) / 12.0, 0.0, 1.0)
    out["weather_wind_stress"] = np.clip((out["weather_wind_speed_mps"] - 8.0) / 10.0, 0.0, 1.0)
    out["weather_comfort_penalty"] = np.clip(
        0.45 * out["weather_heat_stress"]
        + 0.20 * out["weather_cold_stress"]
        + 0.25 * out["weather_rain_flag"]
        + 0.10 * out["weather_wind_stress"],
        0.0,
        1.0,
    )
    out["weather_is_proxy"] = 0.0
    out["weather_source"] = "open-meteo-archive"
    return out.sort_values("datetime").reset_index(drop=True)


def fetch_open_meteo_weather(config: WeatherFetchConfig, timeout: int = 60) -> pd.DataFrame:
    url = open_meteo_archive_url(config)
    with urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return normalize_open_meteo_payload(payload)


def save_weather_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path

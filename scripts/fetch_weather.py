from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beaconnova.config import DEFAULT_DATA_DIR, WEATHER_FILE
from beaconnova.data import load_security_data
from beaconnova.weather_client import BADALING_LATITUDE, BADALING_LONGITUDE, WeatherFetchConfig, fetch_open_meteo_weather, open_meteo_archive_url, save_weather_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Badaling historical weather from Open-Meteo.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Competition data directory.")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path. Defaults to data-dir/weather.csv.")
    parser.add_argument("--start-date", default=None, help="Start date YYYY-MM-DD. Defaults to security data min date.")
    parser.add_argument("--end-date", default=None, help="End date YYYY-MM-DD. Defaults to security data max date.")
    parser.add_argument("--latitude", type=float, default=BADALING_LATITUDE, help="Weather latitude.")
    parser.add_argument("--longitude", type=float, default=BADALING_LONGITUDE, help="Weather longitude.")
    parser.add_argument("--timezone", default="Asia/Shanghai", help="Weather timezone.")
    parser.add_argument("--print-url", action="store_true", help="Print request URL before fetching.")
    return parser.parse_args()


def infer_date_range(data_dir: Path) -> tuple[str, str]:
    security = load_security_data(data_dir)
    dates = security["datetime"].dt.normalize()
    return dates.min().date().isoformat(), dates.max().date().isoformat()


def main() -> None:
    args = parse_args()
    start_date, end_date = infer_date_range(args.data_dir)
    if args.start_date:
        start_date = args.start_date
    if args.end_date:
        end_date = args.end_date
    output = args.output or (args.data_dir / WEATHER_FILE)
    config = WeatherFetchConfig(
        latitude=args.latitude,
        longitude=args.longitude,
        timezone=args.timezone,
        start_date=start_date,
        end_date=end_date,
        output_path=output,
    )
    if args.print_url:
        print(open_meteo_archive_url(config))
    weather = fetch_open_meteo_weather(config)
    save_weather_csv(weather, output)
    print(f"Fetched {len(weather)} hourly weather rows.")
    print(f"Output: {output}")
    print(f"Range: {weather['datetime'].min()} -> {weather['datetime'].max()}")


if __name__ == "__main__":
    main()

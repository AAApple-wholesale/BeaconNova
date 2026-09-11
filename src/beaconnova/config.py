from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_DIR = Path("赛题五基于多源数据融合的文旅场景舒适度评估与风险超前预警模型与分析0805")
DEFAULT_OUTPUT_DIR = Path("outputs") / "baseline"

SECURITY_FILE = "表2安检口每5分钟安检数据：security_gate.csv.csv"
RAIL_FILE = "表3轨道交通客流表：rail_flow.csv"
TICKET_FILE = "表1入园票务数据表：ticket.csv"


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the baseline model."""

    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    horizons: tuple[int, ...] = (6, 12)  # 6/12 five-minute windows = 30/60 minutes
    test_days: int = 21
    random_state: int = 42
    use_ticket_features: bool = True

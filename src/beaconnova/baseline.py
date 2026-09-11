from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class MedianRegressor:
    """Small fallback regressor when sklearn's gradient boosting is unavailable."""

    def __init__(self) -> None:
        self.value_: float = 0.0

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "MedianRegressor":
        self.value_ = float(np.nanmedian(y)) if len(y) else 0.0
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.full(len(x), self.value_, dtype=float)


def make_regressor(random_state: int = 42):
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=random_state,
        )
    except Exception:
        return MedianRegressor()


@dataclass
class TargetModel:
    target: str
    model: object


class MultiTargetBaseline:
    """Train one independent regressor per target column."""

    def __init__(self, target_columns: list[str], random_state: int = 42) -> None:
        self.target_columns = target_columns
        self.random_state = random_state
        self.models: list[TargetModel] = []

    def fit(self, train: pd.DataFrame, feature_cols: list[str]) -> "MultiTargetBaseline":
        self.models = []
        x = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for target in self.target_columns:
            valid = train[target].notna()
            model = make_regressor(self.random_state)
            model.fit(x.loc[valid], train.loc[valid, target])
            self.models.append(TargetModel(target, model))
        return self

    def predict(self, frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        x = frame[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pred = frame[["datetime", "security_gate", "channel", "node_id"]].copy()
        for target_model in self.models:
            values = target_model.model.predict(x)
            values = np.clip(values, a_min=0.0, a_max=None)
            pred["pred_" + target_model.target.removeprefix("target_")] = values
        return pred


def chronological_split(df: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.Series(df["datetime"].dt.normalize().unique()).sort_values()
    if len(dates) <= test_days:
        split_date = dates.iloc[int(len(dates) * 0.8)]
    else:
        split_date = dates.iloc[-test_days]
    train = df[df["datetime"] < split_date].copy()
    test = df[df["datetime"] >= split_date].copy()
    return train, test


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual = pd.to_numeric(actual, errors="coerce")
    predicted = pd.to_numeric(predicted, errors="coerce")
    mask = actual.notna() & predicted.notna()
    if not mask.any():
        return {"mae": np.nan, "rmse": np.nan, "wape": np.nan}
    err = predicted[mask] - actual[mask]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sum(np.abs(actual[mask])))
    wape = float(np.sum(np.abs(err)) / denom) if denom else np.nan
    return {"mae": mae, "rmse": rmse, "wape": wape}


from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .baseline import regression_metrics


@dataclass
class WaitSpecialistResult:
    target: str
    prediction: np.ndarray
    metrics: pd.DataFrame
    feature_importance: pd.DataFrame


def make_wait_regressor(random_state: int = 42):
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=260,
            learning_rate=0.045,
            max_leaf_nodes=15,
            min_samples_leaf=28,
            l2_regularization=0.18,
            early_stopping=True,
            validation_fraction=0.18,
            n_iter_no_change=18,
            random_state=random_state,
        )
    except Exception:
        from .baseline import make_regressor

        return make_regressor(random_state=random_state)


def wait_feature_columns(frame: pd.DataFrame, pred_col: str) -> list[str]:
    preferred = [
        pred_col + "_gcn",
        pred_col + "_tcn_gcn",
        "avg_queue_wait_min",
        "avg_queue_wait_min_lag_1",
        "avg_queue_wait_min_lag_2",
        "avg_queue_wait_min_lag_3",
        "avg_queue_wait_min_lag_6",
        "avg_queue_wait_min_lag_12",
        "avg_queue_wait_min_lag_18",
        "avg_queue_wait_min_lag_24",
        "avg_queue_wait_min_lag_36",
        "avg_queue_wait_min_roll_mean_3",
        "avg_queue_wait_min_roll_mean_6",
        "avg_queue_wait_min_roll_mean_12",
        "avg_queue_wait_min_roll_mean_18",
        "avg_queue_wait_min_roll_mean_24",
        "avg_queue_wait_min_roll_mean_36",
        "avg_queue_wait_min_roll_max_3",
        "avg_queue_wait_min_roll_max_6",
        "avg_queue_wait_min_roll_max_12",
        "avg_queue_wait_min_roll_max_24",
        "avg_queue_wait_min_roll_std_3",
        "avg_queue_wait_min_roll_std_6",
        "avg_queue_wait_min_roll_std_12",
        "avg_queue_wait_min_roll_std_24",
        "wait_diff_1",
        "wait_diff_3",
        "wait_diff_6",
        "wait_momentum_30min",
        "wait_momentum_60min",
        "wait_accel_15min",
        "wait_accel_30min",
        "wait_level_x_momentum_30min",
        "channel_wait_gap",
        "gate_wait_gap",
        "gate_wait_mean",
        "system_wait_mean",
        "gate_wait_roll_mean_6",
        "gate_wait_roll_max_12",
        "system_wait_roll_mean_6",
        "system_wait_roll_max_12",
        "total_person_count",
        "total_person_count_lag_1",
        "total_person_count_lag_6",
        "total_person_count_roll_mean_6",
        "total_person_count_roll_max_12",
        "person_diff_1",
        "person_momentum_30min",
        "total_bag_check_num",
        "total_bag_check_num_lag_1",
        "total_bag_check_num_roll_mean_6",
        "bag_diff_1",
        "bag_per_person",
        "wait_per_person",
        "channel_load_share",
        "gate_system_share",
        "gate_flow_intensity",
        "facility_hourly_pressure",
        "facility_bottleneck_pressure",
        "ticket_facility_hourly_pressure",
        "rail_ropeway_hourly_pressure",
        "rail_ropeway_platform_pressure",
        "ticket_ropeway_pressure",
        "weather_comfort_penalty",
        "weather_heat_stress",
        "weather_cold_stress",
        "weather_rain_flag",
        "weather_wind_stress",
        "hour",
        "dayofweek",
        "slot_5min",
        "sin_slot",
        "cos_slot",
        "is_weekend",
        "is_morning_peak",
        "is_midday_peak",
        "is_closing_period",
    ]
    return [col for col in preferred if col in frame]


def train_wait_specialist(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    target_col: str,
    pred_col: str,
    random_state: int = 42,
) -> WaitSpecialistResult:
    feature_cols = wait_feature_columns(train, pred_col)
    train_x = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    holdout_x = holdout[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_y = train[target_col].astype(float)
    holdout_y = holdout[target_col].astype(float)
    model = make_wait_regressor(random_state=random_state)
    model.fit(train_x, train_y)
    pred = np.clip(model.predict(holdout_x), a_min=0.0, a_max=None)

    metrics = []
    for source_name, source_col in (("gcn", pred_col + "_gcn"), ("tcn_gcn", pred_col + "_tcn_gcn")):
        item = regression_metrics(holdout_y, holdout[source_col].astype(float))
        item.update({"source": source_name, "target": target_col})
        metrics.append(item)
    item = regression_metrics(holdout_y, pd.Series(pred, index=holdout.index))
    item.update({"source": "wait_specialist", "target": target_col})
    metrics.append(item)

    importance = []
    if hasattr(model, "feature_importances_"):
        values = getattr(model, "feature_importances_")
        for name, value in sorted(zip(feature_cols, values), key=lambda item: item[1], reverse=True):
            importance.append({"target": target_col, "feature": name, "importance": float(value)})
    else:
        # HistGradientBoostingRegressor does not expose impurity importances; use simple absolute correlation as a compact diagnostic.
        for name in feature_cols:
            x = pd.to_numeric(train_x[name], errors="coerce").fillna(0.0)
            corr = np.corrcoef(x, train_y)[0, 1] if x.std() > 1e-8 and train_y.std() > 1e-8 else 0.0
            importance.append({"target": target_col, "feature": name, "importance": float(abs(corr)) if np.isfinite(corr) else 0.0})
        importance.sort(key=lambda row: row["importance"], reverse=True)

    return WaitSpecialistResult(
        target=target_col,
        prediction=pred.astype(np.float32),
        metrics=pd.DataFrame(metrics),
        feature_importance=pd.DataFrame(importance),
    )


@dataclass
class WaitSpecialistSource:
    target: str
    train_prediction: np.ndarray
    holdout_prediction: np.ndarray
    metrics: pd.DataFrame
    feature_importance: pd.DataFrame


def fit_wait_specialist_source(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    target_col: str,
    pred_col: str,
    random_state: int = 42,
    folds: int = 4,
) -> WaitSpecialistSource:
    feature_cols = wait_feature_columns(train, pred_col)
    x_all = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_all = train[target_col].astype(float)
    holdout_x = holdout[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    holdout_y = holdout[target_col].astype(float)

    train_pred = np.full(len(train), np.nan, dtype=np.float32)
    fold_edges = np.linspace(0, len(train), folds + 1, dtype=int)
    for fold in range(1, folds + 1):
        start = fold_edges[fold - 1]
        stop = fold_edges[fold]
        if start == 0 or stop <= start:
            continue
        model = make_wait_regressor(random_state=random_state + fold)
        model.fit(x_all.iloc[:start], y_all.iloc[:start])
        train_pred[start:stop] = np.clip(model.predict(x_all.iloc[start:stop]), a_min=0.0, a_max=None)

    fallback = 0.5 * train[pred_col + "_gcn"].to_numpy(dtype=np.float32) + 0.5 * train[pred_col + "_tcn_gcn"].to_numpy(dtype=np.float32)
    train_pred = np.where(np.isfinite(train_pred), train_pred, fallback).astype(np.float32)

    final_model = make_wait_regressor(random_state=random_state)
    final_model.fit(x_all, y_all)
    holdout_pred = np.clip(final_model.predict(holdout_x), a_min=0.0, a_max=None).astype(np.float32)

    metrics = []
    for source_name, values in (
        ("gcn", holdout[pred_col + "_gcn"].to_numpy(dtype=float)),
        ("tcn_gcn", holdout[pred_col + "_tcn_gcn"].to_numpy(dtype=float)),
        ("wait_specialist", holdout_pred),
    ):
        item = regression_metrics(holdout_y, pd.Series(values, index=holdout.index))
        item.update({"source": source_name, "target": target_col})
        metrics.append(item)

    importance = []
    if hasattr(final_model, "feature_importances_"):
        values = getattr(final_model, "feature_importances_")
        for name, value in sorted(zip(feature_cols, values), key=lambda item: item[1], reverse=True):
            importance.append({"target": target_col, "feature": name, "importance": float(value)})
    else:
        for name in feature_cols:
            x = pd.to_numeric(x_all[name], errors="coerce").fillna(0.0)
            corr = np.corrcoef(x, y_all)[0, 1] if x.std() > 1e-8 and y_all.std() > 1e-8 else 0.0
            importance.append({"target": target_col, "feature": name, "importance": float(abs(corr)) if np.isfinite(corr) else 0.0})
        importance.sort(key=lambda row: row["importance"], reverse=True)

    return WaitSpecialistSource(
        target=target_col,
        train_prediction=train_pred,
        holdout_prediction=holdout_pred,
        metrics=pd.DataFrame(metrics),
        feature_importance=pd.DataFrame(importance),
    )

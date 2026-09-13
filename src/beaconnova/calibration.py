from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PredictionCalibrator:
    """Validation-fitted linear post-calibrator for graph forecasts."""

    target_names: list[str]
    feature_names: dict[str, list[str]]
    coefficients: dict[str, np.ndarray]
    ridge_alpha: float = 1e-3


def _candidate_features(target_name: str) -> list[str]:
    if target_name.startswith("person_"):
        return [
            "total_person_count",
            "total_person_count_lag_1",
            "total_person_count_lag_6",
            "total_person_count_lag_12",
            "total_person_count_roll_mean_6",
            "total_person_count_roll_mean_12",
            "gate_person_count",
            "system_person_count",
        ]
    return [
        "avg_queue_wait_min",
        "avg_queue_wait_min_lag_1",
        "avg_queue_wait_min_lag_6",
        "avg_queue_wait_min_lag_12",
        "avg_queue_wait_min_roll_mean_6",
        "avg_queue_wait_min_roll_mean_12",
        "gate_wait_mean",
        "system_wait_mean",
    ]


def _design_matrix(
    pred_target: np.ndarray,
    x: np.ndarray,
    feature_names: list[str],
    security_indices: list[int],
    target_name: str,
) -> tuple[np.ndarray, list[str]]:
    names = ["model_prediction"]
    cols = [pred_target.reshape(-1)]
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    security_x = x[:, security_indices, :]
    for feature in _candidate_features(target_name):
        idx = feature_index.get(feature)
        if idx is None:
            continue
        cols.append(security_x[:, :, idx].reshape(-1))
        names.append(feature)
    cols.append(np.ones_like(cols[0]))
    names.append("intercept")
    design = np.column_stack(cols).astype(np.float64)
    return np.nan_to_num(design, nan=0.0, posinf=0.0, neginf=0.0), names


def fit_prediction_calibrator(
    val_pred: np.ndarray,
    val_y: np.ndarray,
    val_x: np.ndarray,
    feature_names: list[str],
    target_names: list[str],
    security_indices: list[int],
    ridge_alpha: float = 1e-3,
) -> PredictionCalibrator:
    coefficients: dict[str, np.ndarray] = {}
    design_names: dict[str, list[str]] = {}
    for target_idx, target_name in enumerate(target_names):
        design, names = _design_matrix(val_pred[:, :, target_idx], val_x, feature_names, security_indices, target_name)
        target = val_y[:, :, target_idx].reshape(-1).astype(np.float64)
        mask = np.isfinite(target)
        x_fit = design[mask]
        y_fit = target[mask]
        penalty = np.eye(x_fit.shape[1], dtype=np.float64) * ridge_alpha
        penalty[-1, -1] = 0.0
        beta = np.linalg.solve(x_fit.T @ x_fit + penalty, x_fit.T @ y_fit)
        coefficients[target_name] = beta.astype(np.float32)
        design_names[target_name] = names
    return PredictionCalibrator(
        target_names=list(target_names),
        feature_names=design_names,
        coefficients=coefficients,
        ridge_alpha=ridge_alpha,
    )


def apply_prediction_calibrator(
    calibrator: PredictionCalibrator,
    pred: np.ndarray,
    x: np.ndarray,
    feature_names: list[str],
    security_indices: list[int],
) -> np.ndarray:
    out = pred.copy().astype(np.float32)
    for target_idx, target_name in enumerate(calibrator.target_names):
        design, _ = _design_matrix(pred[:, :, target_idx], x, feature_names, security_indices, target_name)
        beta = calibrator.coefficients[target_name].astype(np.float64)
        out[:, :, target_idx] = (design @ beta).reshape(pred.shape[0], pred.shape[1])
    return np.clip(out, a_min=0.0, a_max=None)


def calibration_frame(calibrator: PredictionCalibrator) -> pd.DataFrame:
    rows = []
    for target_name, beta in calibrator.coefficients.items():
        for feature, coef in zip(calibrator.feature_names.get(target_name, []), beta):
            rows.append(
                {
                    "target": target_name,
                    "calibration_feature": feature,
                    "coefficient": float(coef),
                    "ridge_alpha": calibrator.ridge_alpha,
                }
            )
    return pd.DataFrame(rows)


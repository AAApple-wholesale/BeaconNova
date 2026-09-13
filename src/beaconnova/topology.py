from __future__ import annotations

import numpy as np
import pandas as pd

from .graph import GraphSpec, graph_edges_frame


RELATION_PRIORS = {
    "same_security_gate": 0.92,
    "ticket_to_gate": 0.72,
    "rail_to_gate": 0.76,
    "scenic_route": 0.58,
    "front_gate_to_north_route": 0.64,
    "outside_gate_to_south_route": 0.60,
    "ropeway_to_platform": 0.70,
    "rail_to_ropeway": 0.68,
    "ticket_to_ropeway": 0.62,
    "context_to_ticket": 0.25,
    "context_to_rail": 0.25,
    "context_to_security": 0.22,
    "context_to_facility": 0.24,
    "context_to_ropeway": 0.24,
}


def _max_lag_corr(source: pd.Series, target: pd.Series, max_lag: int = 12) -> tuple[float, int]:
    aligned = pd.concat([source, target], axis=1).fillna(0.0)
    if aligned.shape[0] < max_lag + 10:
        return 0.0, 0
    best = 0.0
    best_lag = 0
    for lag in range(0, max_lag + 1):
        shifted = aligned.iloc[:, 0].shift(lag)
        corr = shifted.corr(aligned.iloc[:, 1])
        if pd.notna(corr) and corr > best:
            best = float(corr)
            best_lag = lag
    return max(0.0, best), best_lag


def calibrate_graph_edges(frame: pd.DataFrame, graph: GraphSpec, max_lag: int = 12) -> tuple[np.ndarray, pd.DataFrame]:
    edges = graph_edges_frame(graph)
    node_index = {node_id: idx for idx, node_id in enumerate(graph.node_ids)}
    security_flow = (
        frame.pivot_table(index="datetime", columns="node_id", values="total_person_count", aggfunc="sum")
        .sort_index()
        .fillna(0.0)
    )
    global_signals = frame.groupby("datetime", as_index=True).agg(
        ticket_count_60min=("ticket_count_60min", "mean"),
        rail_passengers_60min=("rail_passengers_60min", "mean"),
        system_person_count=("system_person_count", "mean"),
        weather_comfort_penalty=("weather_comfort_penalty", "mean"),
    ).sort_index().fillna(0.0)

    rows = []
    adjacency = np.eye(len(graph.node_ids), dtype=np.float32)
    for _, row in edges.iterrows():
        source = str(row["source"])
        target = str(row["target"])
        relation = str(row["relation"])
        prior = RELATION_PRIORS.get(relation, 0.35)
        corr = 0.0
        lag = 0
        if source in security_flow and target in security_flow:
            corr, lag = _max_lag_corr(security_flow[source], security_flow[target], max_lag=max_lag)
        elif source == "global:ticket" and target in security_flow:
            corr, lag = _max_lag_corr(global_signals["ticket_count_60min"], security_flow[target], max_lag=max_lag)
        elif source == "global:rail" and target in security_flow:
            corr, lag = _max_lag_corr(global_signals["rail_passengers_60min"], security_flow[target], max_lag=max_lag)
        elif target == "global:ticket" and source in security_flow:
            corr, lag = _max_lag_corr(security_flow[source], global_signals["ticket_count_60min"], max_lag=max_lag)
        elif target == "global:rail" and source in security_flow:
            corr, lag = _max_lag_corr(security_flow[source], global_signals["rail_passengers_60min"], max_lag=max_lag)
        elif source.startswith("doc:") or source.startswith("map:") or target.startswith("doc:") or target.startswith("map:"):
            corr = float(np.clip(global_signals["weather_comfort_penalty"].mean() + 0.1, 0.0, 0.5))
        weight = float(np.clip(0.65 * prior + 0.35 * corr, 0.05, 1.0))
        adjacency[node_index[source], node_index[target]] = weight
        rows.append(
            {
                "source": source,
                "target": target,
                "relation": relation,
                "prior_weight": prior,
                "lag_corr": corr,
                "best_lag_5min": lag,
                "calibrated_weight": weight,
            }
        )
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.power(np.maximum(degree, 1.0), -0.5)
    adjacency = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    return adjacency.astype(np.float32), pd.DataFrame(rows)

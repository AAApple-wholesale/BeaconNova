from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TARGET_TEMPLATE = ["person_h{h}", "wait_h{h}"]
TIME_FEATURES = [
    "hour",
    "minute",
    "dayofweek",
    "month",
    "is_weekend",
    "slot_5min",
    "sin_slot",
    "cos_slot",
    "is_holiday",
    "is_makeup_workday",
    "is_effective_weekend",
    "is_summer_vacation",
    "holiday_seq",
    "days_to_holiday",
    "days_since_holiday",
]
SECURITY_FEATURES = [
    "total_person_count",
    "avg_queue_wait_min",
    "total_bag_check_num",
    "total_person_count_lag_1",
    "total_person_count_lag_2",
    "total_person_count_lag_3",
    "total_person_count_lag_6",
    "total_person_count_lag_12",
    "avg_queue_wait_min_lag_1",
    "avg_queue_wait_min_lag_2",
    "avg_queue_wait_min_lag_3",
    "avg_queue_wait_min_lag_6",
    "avg_queue_wait_min_lag_12",
    "total_bag_check_num_lag_1",
    "total_bag_check_num_lag_6",
    "total_person_count_roll_mean_3",
    "total_person_count_roll_mean_6",
    "total_person_count_roll_mean_12",
    "avg_queue_wait_min_roll_mean_3",
    "avg_queue_wait_min_roll_mean_6",
    "avg_queue_wait_min_roll_mean_12",
    "person_diff_1",
    "wait_diff_1",
    "gate_person_count",
    "gate_wait_mean",
    "gate_bag_count",
    "system_person_count",
    "system_wait_mean",
    "system_bag_count",
    "channel_load_share",
    "gate_system_share",
]
EXOG_FEATURES = [
    "rail_passengers",
    "rail_capacity",
    "rail_load_ratio",
    "rail_passengers_30min",
    "rail_passengers_60min",
    "rail_load_ratio_60min",
    "ticket_count",
    "ticket_count_30min",
    "ticket_count_60min",
    "ticket_count_day_cum",
    "facility_hourly_pressure",
    "facility_instant_pressure",
    "facility_bottleneck_pressure",
    "ticket_facility_hourly_pressure",
    "rail_ropeway_hourly_pressure",
    "rail_ropeway_platform_pressure",
    "ticket_ropeway_pressure",
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
STATIC_FEATURES = [
    "effective_area_sqm",
    "instant_capacity",
    "hourly_capacity",
    "max_dwell_min",
    "weather_exposure_index",
    "bottleneck_risk_index",
    "service_rate_per_min",
    "instant_capacity_share",
    "vehicle_capacity",
    "min_departure_interval_sec",
    "theoretical_hourly_capacity",
    "saturated_hourly_capacity",
    "platform_instant_capacity",
    "queue_limit_min",
    "departures_per_hour",
    "operating_derate_ratio",
    "ropeway_exposure_index",
]
CONTEXT_FEATURES = [
    "asset_file_size_kb",
    "map_width",
    "map_height",
    "map_aspect_ratio",
    "map_brightness_mean",
    "map_brightness_std",
    "doc_char_count",
    "doc_risk_mentions",
    "doc_weather_mentions",
    "doc_ropeway_mentions",
]
NODE_TYPE_FEATURES = ["type_security", "type_facility", "type_ropeway", "type_ticket", "type_rail", "type_map", "type_document"]
GRAPH_FEATURES = TIME_FEATURES + SECURITY_FEATURES + EXOG_FEATURES + STATIC_FEATURES + CONTEXT_FEATURES + NODE_TYPE_FEATURES


@dataclass(frozen=True)
class GraphSpec:
    node_ids: list[str]
    node_names: list[str]
    node_types: list[str]
    security_node_ids: list[str]
    security_indices: list[int]
    edges: list[tuple[str, str, str]]
    adjacency: np.ndarray


@dataclass(frozen=True)
class GraphDataset:
    times: pd.Series
    x: np.ndarray
    y: np.ndarray
    observed: pd.DataFrame
    graph: GraphSpec
    feature_names: list[str]
    target_names: list[str]


def _add_edge(edges: list[tuple[str, str, str]], a: str, b: str, relation: str) -> None:
    if a == b:
        return
    edges.append((a, b, relation))
    edges.append((b, a, relation))


def _node_table(graph: GraphSpec) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "node_index": range(len(graph.node_ids)),
            "node_id": graph.node_ids,
            "node_name": graph.node_names,
            "node_type": graph.node_types,
            "is_supervised_security_node": [idx in set(graph.security_indices) for idx in range(len(graph.node_ids))],
        }
    )


def graph_nodes_frame(graph: GraphSpec) -> pd.DataFrame:
    return _node_table(graph)


def graph_edges_frame(graph: GraphSpec) -> pd.DataFrame:
    rows = []
    for source, target, relation in graph.edges:
        rows.append({"source": source, "target": target, "relation": relation})
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def build_graph_spec(frame: pd.DataFrame, facility_df: pd.DataFrame | None, ropeway_df: pd.DataFrame | None, context_df: pd.DataFrame | None = None) -> GraphSpec:
    security_node_ids = sorted(frame["node_id"].astype(str).unique())
    node_ids = list(security_node_ids)
    node_names = list(security_node_ids)
    node_types = ["security"] * len(node_ids)

    facility_names: list[str] = []
    if facility_df is not None and not facility_df.empty:
        facility_names = facility_df["point_name"].astype(str).tolist()
        for name in facility_names:
            node_ids.append(f"facility:{name}")
            node_names.append(name)
            node_types.append("facility")

    ropeway_names: list[str] = []
    if ropeway_df is not None and not ropeway_df.empty:
        ropeway_names = ropeway_df["ropeway_name"].astype(str).tolist()
        for name in ropeway_names:
            node_ids.append(f"ropeway:{name}")
            node_names.append(name)
            node_types.append("ropeway")

    if context_df is not None and not context_df.empty:
        for _, row in context_df.iterrows():
            node_ids.append(str(row["asset_id"]))
            node_names.append(str(row["asset_name"]))
            node_types.append(str(row["asset_type"]))

    node_ids.extend(["global:ticket", "global:rail"])
    node_names.extend(["票务全局信号", "轨道交通全局信号"])
    node_types.extend(["ticket", "rail"])

    node_set = set(node_ids)
    edges: list[tuple[str, str, str]] = []
    by_gate = frame[["security_gate", "node_id"]].drop_duplicates()
    for _, group in by_gate.groupby("security_gate"):
        ids = group["node_id"].astype(str).tolist()
        for i, source in enumerate(ids):
            for target in ids[i + 1 :]:
                _add_edge(edges, source, target, "same_security_gate")

    for security_id in security_node_ids:
        _add_edge(edges, "global:ticket", security_id, "ticket_to_gate")
        _add_edge(edges, "global:rail", security_id, "rail_to_gate")

    context_node_ids = []
    if context_df is not None and not context_df.empty:
        context_node_ids = context_df["asset_id"].astype(str).tolist()
        for context_id in context_node_ids:
            _add_edge(edges, context_id, "global:ticket", "context_to_ticket")
            _add_edge(edges, context_id, "global:rail", "context_to_rail")
            for security_id in security_node_ids:
                _add_edge(edges, context_id, security_id, "context_to_security")

    north_route = ["北门锁钥关城城楼", "北四楼・好汉石打卡点", "北索道入口平台", "北八楼・好汉坡制高点", "北八至北九 S 弯长城"]
    south_route = ["岔道城古炮展区", "南索道入口平台", "南四楼南线全景机位"]
    for route in (north_route, south_route):
        full = [f"facility:{name}" for name in route if f"facility:{name}" in node_set]
        for source, target in zip(full, full[1:]):
            _add_edge(edges, source, target, "scenic_route")

    for security_id in security_node_ids:
        if security_id.startswith("前山"):
            for name in north_route:
                target = f"facility:{name}"
                if target in node_set:
                    _add_edge(edges, security_id, target, "front_gate_to_north_route")
        else:
            for name in south_route:
                target = f"facility:{name}"
                if target in node_set:
                    _add_edge(edges, security_id, target, "outside_gate_to_south_route")

    for context_id in context_node_ids:
        for name in facility_names:
            target = f"facility:{name}"
            if target in node_set:
                _add_edge(edges, context_id, target, "context_to_facility")
        for ropeway in ropeway_names:
            target = f"ropeway:{ropeway}"
            if target in node_set:
                _add_edge(edges, context_id, target, "context_to_ropeway")

    for ropeway in ropeway_names:
        ropeway_id = f"ropeway:{ropeway}"
        is_north = "北" in ropeway
        route = north_route if is_north else south_route
        for name in route:
            target = f"facility:{name}"
            if target in node_set:
                _add_edge(edges, ropeway_id, target, "ropeway_to_platform")
        _add_edge(edges, "global:rail", ropeway_id, "rail_to_ropeway")
        _add_edge(edges, "global:ticket", ropeway_id, "ticket_to_ropeway")

    index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    adjacency = np.eye(len(node_ids), dtype=np.float32)
    for source, target, _ in edges:
        if source in index and target in index:
            adjacency[index[source], index[target]] = 1.0
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.power(np.maximum(degree, 1.0), -0.5)
    adjacency = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    return GraphSpec(
        node_ids=node_ids,
        node_names=node_names,
        node_types=node_types,
        security_node_ids=security_node_ids,
        security_indices=[index[node_id] for node_id in security_node_ids],
        edges=edges,
        adjacency=adjacency.astype(np.float32),
    )


def _feature_vector() -> dict[str, float]:
    return {name: 0.0 for name in GRAPH_FEATURES}


def _copy_features(target: dict[str, float], row: pd.Series, names: list[str]) -> None:
    for name in names:
        if name in row:
            value = pd.to_numeric(row[name], errors="coerce")
            target[name] = 0.0 if pd.isna(value) else float(value)


def _facility_static(row: pd.Series) -> dict[str, float]:
    out = _feature_vector()
    _copy_features(out, row, [name for name in STATIC_FEATURES if name in row.index])
    out["type_facility"] = 1.0
    return out


def _ropeway_static(row: pd.Series) -> dict[str, float]:
    out = _feature_vector()
    _copy_features(out, row, [name for name in STATIC_FEATURES if name in row.index])
    out["type_ropeway"] = 1.0
    return out


def build_graph_dataset(
    frame: pd.DataFrame,
    facility_df: pd.DataFrame | None,
    ropeway_df: pd.DataFrame | None,
    horizons: tuple[int, ...],
    context_df: pd.DataFrame | None = None,
) -> GraphDataset:
    graph = build_graph_spec(frame, facility_df, ropeway_df, context_df=context_df)
    target_names = []
    target_cols = []
    for horizon in horizons:
        target_names.extend([f"person_h{horizon}", f"wait_h{horizon}"])
        target_cols.extend([f"target_person_h{horizon}", f"target_wait_h{horizon}"])

    valid_frame = frame.dropna(subset=target_cols).copy()
    counts = valid_frame.groupby("datetime")["node_id"].nunique()
    valid_times = counts[counts == len(graph.security_node_ids)].index.sort_values()
    valid_frame = valid_frame[valid_frame["datetime"].isin(valid_times)].copy()
    grouped = {time: group.set_index("node_id", drop=False) for time, group in valid_frame.groupby("datetime", sort=True)}

    facility_lookup = {}
    if facility_df is not None and not facility_df.empty:
        facility_lookup = {f"facility:{row['point_name']}": _facility_static(row) for _, row in facility_df.iterrows()}
    ropeway_lookup = {}
    if ropeway_df is not None and not ropeway_df.empty:
        ropeway_lookup = {f"ropeway:{row['ropeway_name']}": _ropeway_static(row) for _, row in ropeway_df.iterrows()}
    context_lookup = {}
    if context_df is not None and not context_df.empty:
        for _, row in context_df.iterrows():
            features = _feature_vector()
            _copy_features(features, row, CONTEXT_FEATURES)
            if str(row.get("asset_type", "")) == "map":
                features["type_map"] = 1.0
            else:
                features["type_document"] = 1.0
            context_lookup[str(row["asset_id"])] = features

    x = np.zeros((len(valid_times), len(graph.node_ids), len(GRAPH_FEATURES)), dtype=np.float32)
    y = np.zeros((len(valid_times), len(graph.security_indices), len(target_cols)), dtype=np.float32)
    observed_rows = []
    feature_index = {name: idx for idx, name in enumerate(GRAPH_FEATURES)}

    for sample_idx, time in enumerate(valid_times):
        group = grouped[time]
        first = group.iloc[0]
        global_base = _feature_vector()
        _copy_features(global_base, first, TIME_FEATURES + EXOG_FEATURES)
        for node_idx, node_id in enumerate(graph.node_ids):
            features = _feature_vector()
            if node_id in group.index:
                row = group.loc[node_id]
                _copy_features(features, row, TIME_FEATURES + SECURITY_FEATURES + EXOG_FEATURES)
                features["type_security"] = 1.0
                sec_pos = graph.security_node_ids.index(node_id)
                y[sample_idx, sec_pos, :] = row[target_cols].astype(float).to_numpy(dtype=np.float32)
                observed_rows.append(row)
            elif node_id == "global:ticket":
                features.update(global_base)
                features["type_ticket"] = 1.0
            elif node_id == "global:rail":
                features.update(global_base)
                features["type_rail"] = 1.0
            elif node_id in facility_lookup:
                features.update(global_base)
                static = facility_lookup[node_id]
                for key, value in static.items():
                    if value != 0.0:
                        features[key] = value
            elif node_id in ropeway_lookup:
                features.update(global_base)
                static = ropeway_lookup[node_id]
                for key, value in static.items():
                    if value != 0.0:
                        features[key] = value
            elif node_id in context_lookup:
                features.update(global_base)
                static = context_lookup[node_id]
                for key, value in static.items():
                    if value != 0.0:
                        features[key] = value
            for name, value in features.items():
                x[sample_idx, node_idx, feature_index[name]] = value

    observed = pd.DataFrame(observed_rows).reset_index(drop=True)
    return GraphDataset(
        times=pd.Series(valid_times),
        x=np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
        y=np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0),
        observed=observed,
        graph=graph,
        feature_names=list(GRAPH_FEATURES),
        target_names=target_names,
    )


def split_graph_dataset(dataset: GraphDataset, test_days: int) -> tuple[np.ndarray, np.ndarray]:
    dates = pd.Series(dataset.times.dt.normalize().unique()).sort_values()
    if len(dates) <= test_days:
        split_date = dates.iloc[int(len(dates) * 0.8)]
    else:
        split_date = dates.iloc[-test_days]
    train_idx = np.flatnonzero(dataset.times < split_date)
    test_idx = np.flatnonzero(dataset.times >= split_date)
    return train_idx, test_idx






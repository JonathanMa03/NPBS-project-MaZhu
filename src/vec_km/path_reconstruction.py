from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class PathSummary:
    path_id: int
    n_segments: int
    total_length: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    x_span: float
    y_span: float
    monotone_non_decreasing_x: bool
    monotone_non_increasing_y: bool


def _round_point(x: float, y: float, tol: float) -> tuple[float, float]:
    """
    Snap a point to a tolerance-based grid so nearly coincident endpoints can be matched.
    """
    return (round(x / tol) * tol, round(y / tol) * tol)


def add_endpoint_columns(
    segments: pd.DataFrame,
    tol: float = 1e-3,
) -> pd.DataFrame:
    """
    Add snapped endpoint columns to a line-segment DataFrame.

    Parameters
    ----------
    segments : pd.DataFrame
        Must include x0, y0, x1, y1.
    tol : float, default 1e-3
        Snapping tolerance.

    Returns
    -------
    pd.DataFrame
        Copy with p0_key and p1_key columns.
    """
    required = {"x0", "y0", "x1", "y1"}
    missing = required - set(segments.columns)
    if missing:
        raise ValueError(f"Segments missing required columns: {sorted(missing)}")

    out = segments.copy()
    out["p0_key"] = [
        _round_point(x, y, tol) for x, y in zip(out["x0"], out["y0"])
    ]
    out["p1_key"] = [
        _round_point(x, y, tol) for x, y in zip(out["x1"], out["y1"])
    ]
    return out


def build_segment_adjacency(
    segments: pd.DataFrame,
    tol: float = 1e-3,
) -> dict[int, set[int]]:
    """
    Build an undirected adjacency graph among segments that share endpoints.

    Returns
    -------
    dict[int, set[int]]
        Mapping from row index to neighboring row indices.
    """
    segs = add_endpoint_columns(segments, tol=tol)

    endpoint_to_segments: dict[tuple[float, float], list[int]] = defaultdict(list)
    for idx, row in segs.iterrows():
        endpoint_to_segments[row["p0_key"]].append(idx)
        endpoint_to_segments[row["p1_key"]].append(idx)

    adjacency: dict[int, set[int]] = {idx: set() for idx in segs.index}
    for idxs in endpoint_to_segments.values():
        if len(idxs) < 2:
            continue
        for i in idxs:
            for j in idxs:
                if i != j:
                    adjacency[i].add(j)

    return adjacency


def connected_components(
    adjacency: dict[int, set[int]],
) -> list[list[int]]:
    """
    Compute connected components from an adjacency dictionary.
    """
    visited: set[int] = set()
    components: list[list[int]] = []

    for node in adjacency:
        if node in visited:
            continue

        comp: list[int] = []
        queue: deque[int] = deque([node])
        visited.add(node)

        while queue:
            current = queue.popleft()
            comp.append(current)
            for nb in adjacency[current]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

        components.append(comp)

    return components


def reconstruct_paths(
    segments: pd.DataFrame,
    tol: float = 1e-3,
) -> list[pd.DataFrame]:
    """
    Reconstruct connected path components from segment endpoints.

    Returns
    -------
    list[pd.DataFrame]
        List of segment subsets, one per connected path component.
    """
    if segments.empty:
        return []

    adjacency = build_segment_adjacency(segments, tol=tol)
    comps = connected_components(adjacency)

    paths = [segments.loc[comp].copy().reset_index(drop=False) for comp in comps]
    return paths


def summarize_path(path_segments: pd.DataFrame) -> PathSummary:
    """
    Summarize a connected path component.

    Monotonicity checks are heuristic and based on segment centers sorted by x-center.
    """
    if path_segments.empty:
        raise ValueError("Cannot summarize an empty path.")

    x_vals = np.r_[path_segments["x0"].values, path_segments["x1"].values]
    y_vals = np.r_[path_segments["y0"].values, path_segments["y1"].values]

    x_mid = 0.5 * (path_segments["x0"].values + path_segments["x1"].values)
    y_mid = 0.5 * (path_segments["y0"].values + path_segments["y1"].values)

    order = np.argsort(x_mid)
    x_mid_sorted = x_mid[order]
    y_mid_sorted = y_mid[order]

    mono_x = np.all(np.diff(x_mid_sorted) >= -1e-8)
    mono_y_down = np.all(np.diff(y_mid_sorted) <= 1e-8)

    total_length = float(path_segments["length"].sum())

    x_min = float(x_vals.min())
    x_max = float(x_vals.max())
    y_min = float(y_vals.min())
    y_max = float(y_vals.max())

    return PathSummary(
        path_id=-1,
        n_segments=int(len(path_segments)),
        total_length=total_length,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        x_span=x_max - x_min,
        y_span=y_max - y_min,
        monotone_non_decreasing_x=bool(mono_x),
        monotone_non_increasing_y=bool(mono_y_down),
    )


def summarize_paths(
    paths: Iterable[pd.DataFrame],
) -> pd.DataFrame:
    """
    Summarize many reconstructed paths.
    """
    records = []
    for pid, path_df in enumerate(paths):
        summary = summarize_path(path_df)
        summary.path_id = pid
        records.append(summary.__dict__)

    if not records:
        return pd.DataFrame(
            columns=[
                "path_id",
                "n_segments",
                "total_length",
                "x_min",
                "x_max",
                "y_min",
                "y_max",
                "x_span",
                "y_span",
                "monotone_non_decreasing_x",
                "monotone_non_increasing_y",
            ]
        )

    return (
        pd.DataFrame(records)
        .sort_values(["total_length", "n_segments"], ascending=[False, False])
        .reset_index(drop=True)
    )


def select_longest_paths(
    paths: list[pd.DataFrame],
    k: int,
) -> list[pd.DataFrame]:
    """
    Return the k longest reconstructed paths.
    """
    if k <= 0:
        return []

    lengths = [float(path["length"].sum()) for path in paths]
    order = np.argsort(lengths)[::-1][:k]
    return [paths[i] for i in order]


def filter_step_like_paths(
    paths: list[pd.DataFrame],
    min_segments: int = 5,
    require_monotone_x: bool = True,
) -> list[pd.DataFrame]:
    """
    Heuristic filter for KM-like step paths.

    Keeps paths with enough segments and optionally nondecreasing x progression.
    """
    out: list[pd.DataFrame] = []

    for path_df in paths:
        if len(path_df) < min_segments:
            continue

        summary = summarize_path(path_df)

        if require_monotone_x and not summary.monotone_non_decreasing_x:
            continue

        out.append(path_df)

    return out


def reconstruct_and_summarize_paths(
    segments: pd.DataFrame,
    tol: float = 1e-3,
    min_segments: int = 5,
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """
    Convenience wrapper:
    - reconstruct connected paths
    - filter to KM-like paths
    - summarize the filtered paths
    """
    paths = reconstruct_paths(segments, tol=tol)
    paths = filter_step_like_paths(paths, min_segments=min_segments)

    if not paths:
        return paths, summarize_paths([])

    summary_df = summarize_paths(paths)
    return paths, summary_df
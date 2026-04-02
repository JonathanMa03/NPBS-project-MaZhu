from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_segments(segments: pd.DataFrame) -> None:
    required = {"x0", "y0", "x1", "y1", "length", "is_horizontal", "is_vertical"}
    missing = required - set(segments.columns)
    if missing:
        raise ValueError(f"Segments missing required columns: {sorted(missing)}")


def bbox_from_path_summary(path_summary_row: pd.Series) -> dict[str, float]:
    return {
        "x_min": float(path_summary_row["x_min"]),
        "x_max": float(path_summary_row["x_max"]),
        "y_min": float(path_summary_row["y_min"]),
        "y_max": float(path_summary_row["y_max"]),
    }


def detect_axes_from_curve_bbox(
    segments: pd.DataFrame,
    curve_bbox: dict[str, float],
    length_tolerance: float = 0.35,
    proximity_weight: float = 0.01,
) -> dict[str, dict[str, float] | None]:
    """
    Detect x- and y-axes from all segments using the curve bounding box.

    x-axis:
        must be horizontal and close to the bottom of the curve
    y-axis:
        must be vertical and close to the left side of the curve
    """
    _validate_segments(segments)

    x_min = curve_bbox["x_min"]
    x_max = curve_bbox["x_max"]
    y_min = curve_bbox["y_min"]
    y_max = curve_bbox["y_max"]

    target_x_span = x_max - x_min
    target_y_span = y_max - y_min

    # ---- X axis: horizontal only ----
    horiz = segments.loc[segments["is_horizontal"]].copy()
    if not horiz.empty:
        horiz["seg_x_min"] = horiz[["x0", "x1"]].min(axis=1)
        horiz["seg_x_max"] = horiz[["x0", "x1"]].max(axis=1)
        horiz["seg_y"] = 0.5 * (horiz["y0"] + horiz["y1"])
        horiz["x_span"] = horiz["seg_x_max"] - horiz["seg_x_min"]

        horiz["length_rel_err"] = np.abs(horiz["x_span"] - target_x_span) / max(target_x_span, 1e-8)
        horiz["y_dist"] = np.abs(horiz["seg_y"] - y_max)

        horiz = horiz.loc[horiz["length_rel_err"] <= length_tolerance].copy()
        if not horiz.empty:
            horiz["score"] = horiz["length_rel_err"] + proximity_weight * horiz["y_dist"]
            horiz = horiz.sort_values(["score", "length"], ascending=[True, False]).reset_index()

    # ---- Y axis: vertical only ----
    vert = segments.loc[segments["is_vertical"]].copy()
    if not vert.empty:
        vert["seg_y_min"] = vert[["y0", "y1"]].min(axis=1)
        vert["seg_y_max"] = vert[["y0", "y1"]].max(axis=1)
        vert["seg_x"] = 0.5 * (vert["x0"] + vert["x1"])
        vert["y_span"] = vert["seg_y_max"] - vert["seg_y_min"]

        vert["length_rel_err"] = np.abs(vert["y_span"] - target_y_span) / max(target_y_span, 1e-8)
        vert["x_dist"] = np.abs(vert["seg_x"] - x_min)

        vert = vert.loc[vert["length_rel_err"] <= length_tolerance].copy()
        if not vert.empty:
            vert["score"] = vert["length_rel_err"] + proximity_weight * vert["x_dist"]
            vert = vert.sort_values(["score", "length"], ascending=[True, False]).reset_index()

    out = {"x_axis": None, "y_axis": None}

    if "horiz" in locals() and not horiz.empty:
        best = horiz.iloc[0]
        out["x_axis"] = {
            "segment_index": int(best["index"]),
            "x0": float(best["x0"]),
            "y0": float(best["y0"]),
            "x1": float(best["x1"]),
            "y1": float(best["y1"]),
            "score": float(best["score"]),
        }

    if "vert" in locals() and not vert.empty:
        best = vert.iloc[0]
        out["y_axis"] = {
            "segment_index": int(best["index"]),
            "x0": float(best["x0"]),
            "y0": float(best["y0"]),
            "x1": float(best["x1"]),
            "y1": float(best["y1"]),
            "score": float(best["score"]),
        }

    return out


def axis_dataframe_from_detection(
    detection: dict[str, dict[str, float] | None],
) -> pd.DataFrame:
    records = []

    for axis_name in ["x_axis", "y_axis"]:
        axis = detection.get(axis_name)
        if axis is None:
            records.append(
                {
                    "axis": axis_name,
                    "segment_index": None,
                    "x0": None,
                    "y0": None,
                    "x1": None,
                    "y1": None,
                    "score": None,
                }
            )
        else:
            records.append({"axis": axis_name, **axis})

    return pd.DataFrame(records)
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def plot_segments(
    segments: pd.DataFrame,
    ax: plt.Axes | None = None,
    color: str = "steelblue",
    linewidth: float = 1.0,
    alpha: float = 0.7,
    invert_y: bool = True,
    title: str | None = None,
) -> plt.Axes:
    """
    Plot raw line segments from a PDF page.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    for _, row in segments.iterrows():
        ax.plot([row["x0"], row["x1"]], [row["y0"], row["y1"]],
                color=color, linewidth=linewidth, alpha=alpha)

    if invert_y:
        ax.invert_yaxis()

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    return ax


def plot_paths(
    paths: Iterable[pd.DataFrame],
    ax: plt.Axes | None = None,
    linewidth: float = 2.0,
    invert_y: bool = True,
    title: str | None = None,
) -> plt.Axes:
    """
    Plot reconstructed path components in different colors.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    cmap = plt.get_cmap("tab10")
    for i, path_df in enumerate(paths):
        color = cmap(i % 10)
        for _, row in path_df.iterrows():
            ax.plot([row["x0"], row["x1"]], [row["y0"], row["y1"]],
                    color=color, linewidth=linewidth, alpha=0.9)

    if invert_y:
        ax.invert_yaxis()

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    return ax


def overlay_axes(
    ax: plt.Axes,
    axes_detection: dict[str, dict[str, float] | None],
    x_color: str = "red",
    y_color: str = "green",
) -> plt.Axes:
    """
    Overlay detected x- and y-axes on an existing plot.
    """
    x_axis = axes_detection.get("x_axis")
    y_axis = axes_detection.get("y_axis")

    if x_axis is not None:
        ax.plot([x_axis["x0"], x_axis["x1"]],
                [x_axis["y0"], x_axis["y1"]],
                color=x_color, linewidth=3, label="x_axis")

    if y_axis is not None:
        ax.plot([y_axis["x0"], y_axis["x1"]],
                [y_axis["y0"], y_axis["y1"]],
                color=y_color, linewidth=3, label="y_axis")

    ax.legend()
    return ax


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 200) -> Path:
    """
    Save a matplotlib figure to disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
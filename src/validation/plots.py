from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


def plot_km_comparison(
    df_true: pd.DataFrame,
    df_recon: pd.DataFrame,
    time_col: str = "time",
    event_col: str = "event",
    title: str = "KM Curve: True vs Reconstructed",
    figsize: tuple[int, int] = (7, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Overlay Kaplan–Meier curves from true and reconstructed IPD.
    """
    km_true = KaplanMeierFitter()
    km_recon = KaplanMeierFitter()

    km_true.fit(df_true[time_col], event_observed=df_true[event_col], label="True")
    km_recon.fit(df_recon[time_col], event_observed=df_recon[event_col], label="Reconstructed")

    fig, ax = plt.subplots(figsize=figsize)
    km_true.plot_survival_function(ax=ax)
    km_recon.plot_survival_function(ax=ax)

    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.set_title(title)
    return fig, ax


def plot_survival_difference_curve(
    df_true: pd.DataFrame,
    df_recon: pd.DataFrame,
    time_col: str = "time",
    event_col: str = "event",
    n_grid: int = 100,
    title: str = "Reconstructed minus True Survival",
    figsize: tuple[int, int] = (7, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot S_reconstructed(t) - S_true(t) over a common time grid.
    """
    km_true = KaplanMeierFitter()
    km_recon = KaplanMeierFitter()

    km_true.fit(df_true[time_col], event_observed=df_true[event_col], label="True")
    km_recon.fit(df_recon[time_col], event_observed=df_recon[event_col], label="Reconstructed")

    t_max = min(df_true[time_col].max(), df_recon[time_col].max())
    grid = np.linspace(0, t_max, n_grid)

    s_true = km_true.survival_function_at_times(grid).values
    s_recon = km_recon.survival_function_at_times(grid).values
    diff = s_recon - s_true

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(grid, diff, linewidth=2)
    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_xlabel("Time")
    ax.set_ylabel("S_reconstructed(t) - S_true(t)")
    ax.set_title(title)
    return fig, ax


def plot_risk_table_comparison(
    risk_comparison: pd.DataFrame,
    time_col: str = "time",
    target_col: str = "n_risk_target",
    recon_col: str = "n_risk_reconstructed",
    title: str = "Risk Table Comparison",
    figsize: tuple[int, int] = (7, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot target vs reconstructed number-at-risk over time.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(risk_comparison[time_col], risk_comparison[target_col], marker="o", label="Target")
    ax.plot(risk_comparison[time_col], risk_comparison[recon_col], marker="s", label="Reconstructed")

    ax.set_xlabel("Time")
    ax.set_ylabel("Number at risk")
    ax.set_title(title)
    ax.legend()
    return fig, ax


def plot_interval_error(
    event_table: pd.DataFrame,
    time_col: str = "time",
    error_col: str = "interval_error",
    title: str = "Interval Reconstruction Error",
    figsize: tuple[int, int] = (7, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot interval-level survival reconstruction error if available.
    """
    if error_col not in event_table.columns:
        raise ValueError(f"event_table does not contain column '{error_col}'.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(event_table[time_col], event_table[error_col], marker="o", linewidth=1.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("Absolute interval error")
    ax.set_title(title)
    return fig, ax


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    dpi: int = 200,
    close: bool = True,
) -> Path:
    """
    Save a figure to disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


@dataclass
class KMRenderResult:
    curve_points: pd.DataFrame
    censor_points: pd.DataFrame
    event_table: pd.DataFrame


def km_from_ipd(
    time: np.ndarray,
    event: np.ndarray,
    label: str = "KM",
) -> KaplanMeierFitter:
    """
    Fit a Kaplan-Meier curve from IPD.
    """
    kmf = KaplanMeierFitter()
    kmf.fit(time, event_observed=event, label=label)
    return kmf


def extract_step_points(
    kmf: KaplanMeierFitter,
) -> pd.DataFrame:
    """
    Extract KM step coordinates from a fitted KaplanMeierFitter.

    Returns
    -------
    pd.DataFrame
        Columns: time, survival
    """
    sf = kmf.survival_function_.reset_index()
    sf.columns = ["time", "survival"]
    return sf


def extract_censor_points(
    kmf: KaplanMeierFitter,
) -> pd.DataFrame:
    """
    Extract censoring coordinates from a fitted KaplanMeierFitter.

    Returns
    -------
    pd.DataFrame
        Columns: time, survival
    """
    et = kmf.event_table.reset_index().rename(columns={"event_at": "time"})
    cens_times = et.loc[et["censored"] > 0, "time"].values

    if len(cens_times) == 0:
        return pd.DataFrame(columns=["time", "survival", "n_censored"])

    surv_at_censor = kmf.survival_function_at_times(cens_times).values
    n_censored = et.loc[et["censored"] > 0, "censored"].values

    return pd.DataFrame(
        {
            "time": cens_times,
            "survival": surv_at_censor,
            "n_censored": n_censored,
        }
    )


def render_km_plot(
    curve_points: pd.DataFrame,
    censor_points: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "Kaplan–Meier Curve",
    figsize: tuple[int, int] = (7, 5),
) -> Path | None:
    """
    Render a KM curve and censor marks to an image file.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.step(
        curve_points["time"],
        curve_points["survival"],
        where="post",
        linewidth=2,
        label="KM curve",
    )

    if not censor_points.empty:
        ax.scatter(
            censor_points["time"],
            censor_points["survival"],
            marker="|",
            s=100,
            label="Censor marks",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.05)

    if output_path is None:
        plt.show()
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_km_render_from_ipd(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    label: str = "KM",
) -> KMRenderResult:
    """
    Fit KM from IPD and return curve points, censor points, and event table.
    """
    kmf = km_from_ipd(
        time=df[time_col].to_numpy(),
        event=df[event_col].to_numpy(),
        label=label,
    )

    curve_points = extract_step_points(kmf)
    censor_points = extract_censor_points(kmf)
    event_table = kmf.event_table.reset_index()

    return KMRenderResult(
        curve_points=curve_points,
        censor_points=censor_points,
        event_table=event_table,
    )
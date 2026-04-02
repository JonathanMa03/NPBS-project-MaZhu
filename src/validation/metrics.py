from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


def km_from_ipd(df: pd.DataFrame, time_col: str, event_col: str):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    return kmf


def survival_at_times(kmf, times: np.ndarray):
    return kmf.survival_function_at_times(times).values


def km_curve_error(
    df_true: pd.DataFrame,
    df_recon: pd.DataFrame,
    time_col: str = "time",
    event_col: str = "event",
    n_grid: int = 100,
) -> float:
    """
    RMSE between true and reconstructed KM curves
    """
    km_true = km_from_ipd(df_true, time_col, event_col)
    km_recon = km_from_ipd(df_recon, time_col, event_col)

    t_max = min(df_true[time_col].max(), df_recon[time_col].max())
    grid = np.linspace(0, t_max, n_grid)

    S_true = survival_at_times(km_true, grid)
    S_recon = survival_at_times(km_recon, grid)

    return float(np.sqrt(np.mean((S_true - S_recon) ** 2)))


def survival_difference_at_t(
    df_true: pd.DataFrame,
    df_recon: pd.DataFrame,
    t0: float,
    time_col: str = "time",
    event_col: str = "event",
) -> float:
    """
    Difference in survival at time t0
    """
    km_true = km_from_ipd(df_true, time_col, event_col)
    km_recon = km_from_ipd(df_recon, time_col, event_col)

    S_true = float(km_true.predict(t0))
    S_recon = float(km_recon.predict(t0))

    return S_recon - S_true


def summary_metrics(
    df_true: pd.DataFrame,
    df_recon: pd.DataFrame,
    t0: float = 10.0,
) -> dict:
    """
    Collect key reconstruction metrics
    """
    return {
        "km_rmse": km_curve_error(df_true, df_recon),
        "survival_diff_t0": survival_difference_at_t(df_true, df_recon, t0),
        "n_true": len(df_true),
        "n_recon": len(df_recon),
    }
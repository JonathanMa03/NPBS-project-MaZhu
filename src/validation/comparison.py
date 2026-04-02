from __future__ import annotations

import pandas as pd

from src.validation.metrics import summary_metrics
from src.validation.plots import plot_km_comparison


def compare_reconstructions(
    df_true: pd.DataFrame,
    ipd_ours: pd.DataFrame,
    ipd_resolve: pd.DataFrame,
    t0: float = 10.0,
) -> pd.DataFrame:
    """
    Compare reconstruction methods.

    Returns a tidy DataFrame with metrics for:
    - our method
    - RESOLVE-IPD
    """
    metrics_ours = summary_metrics(df_true, ipd_ours, t0=t0)
    metrics_resolve = summary_metrics(df_true, ipd_resolve, t0=t0)

    df = pd.DataFrame([
        {"method": "ours", **metrics_ours},
        {"method": "resolve_ipd", **metrics_resolve},
    ])

    return df


def compare_and_plot(
    df_true: pd.DataFrame,
    ipd_ours: pd.DataFrame,
    ipd_resolve: pd.DataFrame,
    t0: float = 10.0,
):
    """
    Full comparison:
    - metrics table
    - KM plots
    """
    results = compare_reconstructions(
        df_true=df_true,
        ipd_ours=ipd_ours,
        ipd_resolve=ipd_resolve,
        t0=t0,
    )

    print("=== Reconstruction Comparison ===")
    print(results)

    # Plot ours vs true
    fig1, _ = plot_km_comparison(df_true, ipd_ours, title="Our Method vs True")

    # Plot resolve vs true
    fig2, _ = plot_km_comparison(df_true, ipd_resolve, title="RESOLVE-IPD vs True")

    return results, fig1, fig2
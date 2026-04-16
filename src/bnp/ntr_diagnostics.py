from __future__ import annotations

from typing import Iterable

import pandas as pd

from .ntr import fit_ntr_piecewise, compute_delta_posterior


def run_grid_sensitivity(
    df,
    interval_grid: Iterable[int] = (6, 8, 10, 12),
    prior_shapes: Iterable[float] = (0.5, 1.0),
    prior_rates: Iterable[float] = (0.5, 1.0),
    t0: float = 12.0,
    random_state: int = 733,
) -> pd.DataFrame:
    """
    Run a small sensitivity study for the NTR model.
    """
    rows = []
    for J in interval_grid:
        for a in prior_shapes:
            for b in prior_rates:
                fit = fit_ntr_piecewise(
                    df=df,
                    n_intervals=J,
                    prior_shape=a,
                    prior_rate=b,
                    t0_default=t0,
                    random_state=random_state,
                )
                post = compute_delta_posterior(fit, t0=t0)
                rows.append({
                    "n_intervals": J,
                    "prior_shape": a,
                    "prior_rate": b,
                    "Delta_mean": post["mean"],
                    "Delta_sd": post["sd"],
                    "q025": post["q025"],
                    "q975": post["q975"],
                })
    return pd.DataFrame(rows)
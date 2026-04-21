from __future__ import annotations

import numpy as np
import pandas as pd


def _logit(p: float) -> float:
    p = np.clip(float(p), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def simulate_covariates_from_aux(
    df: pd.DataFrame,
    age_sd: float = 8.0,
    stage_sd: float = 0.25,
    random_state: int = 733,
) -> pd.DataFrame:
    """
    Simulate individual-level covariates using merged auxiliary summaries.

    Expected columns in df:
    - age_median
    - male_rate
    - ecog0_rate
    - metastatic_rate
    - biomarker_rate

    Output adds:
    - age
    - male
    - ecog0
    - metastatic
    - biomarker
    - stage

    Notes
    -----
    - age is approximated using a normal distribution centered at age_median
    - stage is a severity proxy constructed from metastatic status + ecog
    """
    required = {
        "age_median",
        "male_rate",
        "ecog0_rate",
        "metastatic_rate",
        "biomarker_rate",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required auxiliary columns: {sorted(missing)}")

    out = df.copy()
    rng = np.random.default_rng(random_state)

    n = len(out)

    # Age
    out["age"] = rng.normal(loc=out["age_median"].to_numpy(), scale=age_sd, size=n)

    # Binary covariates
    out["male"] = rng.binomial(1, out["male_rate"].to_numpy(), size=n)
    out["ecog0"] = rng.binomial(1, out["ecog0_rate"].to_numpy(), size=n)
    out["metastatic"] = rng.binomial(1, out["metastatic_rate"].to_numpy(), size=n)
    out["biomarker"] = rng.binomial(1, out["biomarker_rate"].to_numpy(), size=n)

    # Simple stage / severity proxy:
    # higher if metastatic and lower if ECOG 0
    stage_base = (
        1.2
        + 0.8 * out["metastatic"].to_numpy()
        + 0.4 * (1 - out["ecog0"].to_numpy())
        + 0.2 * out["biomarker"].to_numpy()
    )
    out["stage"] = stage_base + rng.normal(0.0, stage_sd, size=n)

    return out


def summarize_simulated_covariates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize simulated covariates by trial and subgroup.
    """
    group_cols = ["trial_id", "subgroup"]
    summary = (
        df.groupby(group_cols)
        .agg(
            age_median=("age", "median"),
            age_mean=("age", "mean"),   # optional: keep if you want both
            male_rate=("male", "mean"),
            ecog0_rate=("ecog0", "mean"),
            metastatic_rate=("metastatic", "mean"),
            biomarker_rate=("biomarker", "mean"),
            stage_mean=("stage", "mean"),
            n=("time", "size"),
        )
        .reset_index()
    )
    return summary
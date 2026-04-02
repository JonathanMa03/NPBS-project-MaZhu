from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TrialSimulationResult:
    ipd: pd.DataFrame
    summary: pd.DataFrame


def simulate_baseline_covariates(
    n: int,
    rng: np.random.Generator,
    age_mean: float = 60.0,
    age_sd: float = 10.0,
    biomarker_prob: float = 0.5,
    stage_probs: tuple[float, float, float] = (0.4, 0.4, 0.2),
) -> pd.DataFrame:
    age = rng.normal(loc=age_mean, scale=age_sd, size=n)
    biomarker = rng.binomial(1, biomarker_prob, size=n)

    stage_values = np.array([1, 2, 3])
    stage = rng.choice(stage_values, size=n, p=stage_probs)

    return pd.DataFrame(
        {
            "age": age,
            "biomarker": biomarker,
            "stage": stage,
        }
    )


def simulate_treatment_assignment(
    n: int,
    rng: np.random.Generator,
    treatment_prob: float = 0.5,
) -> np.ndarray:
    return rng.binomial(1, treatment_prob, size=n)


def simulate_event_times_exponential(
    treatment: np.ndarray,
    covariates: pd.DataFrame,
    rng: np.random.Generator,
    baseline_rate: float = 0.08,
    beta_treatment: float = -0.35,
    beta_age: float = 0.02,
    beta_biomarker: float = -0.25,
    beta_stage2: float = 0.25,
    beta_stage3: float = 0.55,
) -> np.ndarray:
    age_centered = covariates["age"].to_numpy() - covariates["age"].mean()
    biomarker = covariates["biomarker"].to_numpy()
    stage = covariates["stage"].to_numpy()

    lp = (
        beta_treatment * treatment
        + beta_age * age_centered
        + beta_biomarker * biomarker
        + beta_stage2 * (stage == 2).astype(float)
        + beta_stage3 * (stage == 3).astype(float)
    )

    rates = baseline_rate * np.exp(lp)
    return rng.exponential(scale=1.0 / rates, size=len(treatment))


def simulate_censoring_times(
    n: int,
    rng: np.random.Generator,
    admin_time: float = 24.0,
    early_censor_prob: float = 0.15,
    early_censor_max: float = 12.0,
) -> np.ndarray:
    is_early = rng.binomial(1, early_censor_prob, size=n).astype(bool)
    censor = np.full(n, admin_time, dtype=float)
    censor[is_early] = rng.uniform(0.0, early_censor_max, size=is_early.sum())
    return censor


def simulate_single_trial(
    n: int = 300,
    seed: int = 123,
    trial_id: str = "trial_001",
    age_mean: float = 60.0,
    age_sd: float = 10.0,
    biomarker_prob: float = 0.5,
    stage_probs: tuple[float, float, float] = (0.4, 0.4, 0.2),
    treatment_prob: float = 0.5,
    baseline_rate: float = 0.08,
    beta_treatment: float = -0.35,
    beta_age: float = 0.02,
    beta_biomarker: float = -0.25,
    beta_stage2: float = 0.25,
    beta_stage3: float = 0.55,
    admin_time: float = 24.0,
    early_censor_prob: float = 0.15,
    early_censor_max: float = 12.0,
) -> TrialSimulationResult:
    rng = np.random.default_rng(seed)

    X = simulate_baseline_covariates(
        n=n,
        rng=rng,
        age_mean=age_mean,
        age_sd=age_sd,
        biomarker_prob=biomarker_prob,
        stage_probs=stage_probs,
    )

    A = simulate_treatment_assignment(n=n, rng=rng, treatment_prob=treatment_prob)

    T_event = simulate_event_times_exponential(
        treatment=A,
        covariates=X,
        rng=rng,
        baseline_rate=baseline_rate,
        beta_treatment=beta_treatment,
        beta_age=beta_age,
        beta_biomarker=beta_biomarker,
        beta_stage2=beta_stage2,
        beta_stage3=beta_stage3,
    )

    T_censor = simulate_censoring_times(
        n=n,
        rng=rng,
        admin_time=admin_time,
        early_censor_prob=early_censor_prob,
        early_censor_max=early_censor_max,
    )

    time = np.minimum(T_event, T_censor)
    event = (T_event <= T_censor).astype(int)

    ipd = X.copy()
    ipd["treatment"] = A
    ipd["time"] = time
    ipd["event"] = event
    ipd["trial_id"] = trial_id

    summary = pd.DataFrame(
        {
            "trial_id": [trial_id],
            "n": [n],
            "n_events": [int(event.sum())],
            "event_rate": [float(event.mean())],
            "mean_age": [float(ipd["age"].mean())],
            "biomarker_rate": [float(ipd["biomarker"].mean())],
            "stage1_rate": [float((ipd["stage"] == 1).mean())],
            "stage2_rate": [float((ipd["stage"] == 2).mean())],
            "stage3_rate": [float((ipd["stage"] == 3).mean())],
            "treatment_rate": [float(ipd["treatment"].mean())],
        }
    )

    return TrialSimulationResult(ipd=ipd, summary=summary)


def simulate_multi_trial(
    n_trials: int = 4,
    n_per_trial: int = 300,
    seed: int = 123,
) -> TrialSimulationResult:
    rng = np.random.default_rng(seed)

    ipd_list = []
    summary_list = []

    for s in range(n_trials):
        trial_seed = int(rng.integers(1, 10_000_000))
        trial_id = f"trial_{s+1:03d}"

        age_mean = 58.0 + 4.0 * rng.normal()
        biomarker_prob = float(np.clip(0.5 + 0.1 * rng.normal(), 0.2, 0.8))
        baseline_rate = float(np.clip(0.08 + 0.02 * rng.normal(), 0.03, 0.20))

        stage_raw = np.abs(np.array([0.4, 0.4, 0.2]) + 0.05 * rng.normal(size=3))
        stage_probs = tuple((stage_raw / stage_raw.sum()).tolist())

        result = simulate_single_trial(
            n=n_per_trial,
            seed=trial_seed,
            trial_id=trial_id,
            age_mean=age_mean,
            biomarker_prob=biomarker_prob,
            stage_probs=stage_probs,
            baseline_rate=baseline_rate,
        )

        ipd_list.append(result.ipd)
        summary_list.append(result.summary)

    ipd = pd.concat(ipd_list, ignore_index=True)
    summary = pd.concat(summary_list, ignore_index=True)

    return TrialSimulationResult(ipd=ipd, summary=summary)
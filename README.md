# Causal Survival Probability Estimation after Reconstructed IPD using Auxiliary Summary Information

Authors: [Jonathan Ma](https://jonathanma03.github.io/), [Sijia Zhu](https://github.com/ChesleaZ)

Mentors: [Dr. Yanxun Xu](https://www.ams.jhu.edu/~yxu70/), [Lang Lang](https://scholar.google.com/citations?user=BUyRyUIAAAAJ&hl=en)

---

## Aim

This project develops a framework for estimating causal survival probabilities using reconstructed individual patient data (IPD) from published Kaplan–Meier curves, augmented with auxiliary summary information such as baseline characteristics reported in Table 1.

Within a single randomized trial, causal survival effects are identifiable under the original randomization scheme. However, when pooling reconstructed IPD across multiple studies, this identification no longer holds due to heterogeneity in baseline covariate distributions and study populations. In such settings, auxiliary paper-level summaries provide partial information about the underlying covariate structure, but do not fully determine it.

We propose a Bayesian nonparametric approach that models the unknown covariate distribution (or equivalently, reweighting scheme) using a flexible prior constrained by auxiliary summary statistics. This allows causal survival estimands—specifically treatment-specific survival probabilities at fixed time points—to be estimated while propagating uncertainty arising from both IPD reconstruction and incomplete covariate information.

---

## Keywords

Bayesian nonparametrics, causal survival analysis, IPD reconstruction, Kaplan–Meier, Dirichlet process, summary statistics, transportability, reweighting, oncology trials

---

## Data Source

The project operates on reconstructed IPD derived from published Kaplan–Meier curves using established reconstruction methods (e.g., RESOLVE-IPD). The reconstructed dataset contains:
- Event or censoring times
- Event indicators
- Treatment assignments

In addition, auxiliary summary statistics extracted from trial publications are used, including:
- Baseline covariate summaries (e.g., mean age, gender proportions)
- Subgroup-level information where available
- Study-level characteristics

For simulation studies, synthetic multi-study datasets are generated with known covariate distributions and survival mechanisms, allowing evaluation of bias, variance, and coverage under controlled violations of identifiability.

---

## Methodology

### 1. Problem Setup

Let $T$ denote survival time, $A \in {0,1}$ treatment assignment, and $X$ baseline covariates. The target estimand is the causal survival probability at a fixed time point $t_0$:

$$
S^a(t_0) = \Pr(T^a > t_0), \quad a \in {0,1},
$$

with contrast:

$$
\Delta(t_0) = S^1(t_0) - S^0(t_0).
$$

After pooling reconstructed IPD across studies, the distribution of $X$ is not observed and cannot be recovered from individual-level data alone.

### 2. Bayesian Nonparametric Reweighting

We model the unknown target covariate distribution implicitly through a set of weights $w = (w_1, \dots, w_n)$ assigned to reconstructed individuals:

$$
w \sim \text{Dirichlet}(\alpha).
$$

These weights represent a flexible, nonparametric distribution over the empirical support of the reconstructed data.

To incorporate auxiliary summary information, we impose constraints such that weighted covariate summaries match reported values:

$$
\sum_i w_i X_i \approx \bar{X}_{\text{reported}}.
$$

This is implemented via a pseudo-likelihood or soft constraint, allowing uncertainty around reported summaries.

### 3. Estimation of Causal Survival

Given weights $w$, treatment-specific survival probabilities are estimated as:

$$
S^a(t_0) = \sum_{i: A_i = a} w_i \cdot \mathbf{1}(T_i > t_0).
$$

Posterior inference integrates over the distribution of weights, yielding:
- posterior mean estimates of $S^a(t_0)$
- credible intervals reflecting uncertainty in covariate structure


### 4. Simulation Study

A multi-study simulation framework is used to evaluate performance:
- Studies with heterogeneous covariate distributions
- Treatment randomized within each study
- Only aggregate summaries available for pooled inference

Methods compared:
- Naive pooled estimator (ignores covariate shift)
- Parametric reweighting (e.g., linear or entropy balancing)
- Proposed BNP reweighting approach

Evaluation metrics:
- Bias
- Root mean squared error
- Coverage of credible intervals

### 5. Extensions (Post-Project)

Planned extensions include:
- Dependent Dirichlet process models for study-specific distributions
- Joint modeling of reconstruction uncertainty and causal inference
- Full survival-curve inference beyond fixed time points
- Application to real multi-trial oncology datasets

---

## File Structure

```text
project/
│
├── data/
│   ├── raw/                             # original trial PDFs, SVGs, extracted tables, source registry
│   ├── interim/                         # intermediate RESOLVE-IPD artifacts (VEC-KM, CEN-KM, MAPLE)
│   ├── processed/                       # reconstructed IPD, subgroup ensembles, pooled analysis datasets
│   └── outputs/                         # posterior traces, summaries, simulation results, final derived outputs
│
├── docs/
│   ├── CHANGELOG.md
│   ├── Python_git.md
│   └── R_git.md
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_vec_km_extraction.ipynb
│   ├── 03_cen_km_reconstruction.ipynb
│   ├── 04_maple_subgroup_recovery.ipynb
│   ├── 05_reconstruction_validation.ipynb
│   ├── 06_trial_level_survival_estimands.ipynb
│   ├── 07_pooled_causal_survival_with_auxiliary_summaries.ipynb
│   ├── 08_bnp_extension.ipynb
│   ├── 09_simulation_study.ipynb
│   └── 10_results_and_discussion.ipynb
│
├── external/
│   └── resolve_ipd/         # Original RESOLVE-IPD for comparison
├── src/
│   ├── acquisition/                    # source registries, trial metadata, file/path utilities
│   ├── vec_km/                         # vector KM extraction: paths, axes, censor marks, coordinate transforms
│   ├── cen_km/                         # censor-aware IPD reconstruction and risk-table alignment
│   ├── maple/                          # subgroup label recovery, optimization, refinement, ensembles
│   ├── validation/                     # reconstruction checks, summary-stat comparison, RMSE / diagnostics
│   ├── survival/                       # KM, milestone survival, RMST, hazard summaries
│   ├── causal/                         # pooled causal estimands, weighting, standardization, uncertainty propagation
│   ├── bnp/                            # Bayesian nonparametric extension: Dirichlet / DP weighting and posterior inference
│   ├── simulation/                     # synthetic trials, survival generation, summary-stat generation, scenarios
│   ├── utils/                          # generic helpers: plotting, serialization, logging, math, paths
│   └── config.py                       # project-wide constants and default settings
│
├── reports/
│   ├── figures/                        # exported figures for paper and slides
│   ├── tables/                         # exported summary tables
│   ├── presentation.pdf
│   └── final_report.pdf
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

Lang, L., et al. (2024). RESOLVE-IPD: A high-fidelity approach to reconstructing individual patient data from Kaplan–Meier curves with subgroup information. [Link](https://arxiv.org/pdf/2511.01785)

Ying, S., et al. (2025). Summary-statistics-based causal inference under covariate shift: methods and applications [Link](https://arxiv.org/pdf/2603.02474v1)
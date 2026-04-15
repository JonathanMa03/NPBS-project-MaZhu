# Causal Survival Probability Estimation after Reconstructed IPD using Auxiliary Summary Information

Authors: [Jonathan Ma](https://jonathanma03.github.io/), [Sijia Zhu](https://github.com/ChesleaZ)

Mentors: [Dr. Yanxun Xu](https://www.ams.jhu.edu/~yxu70/), [Lang Lang](https://scholar.google.com/citations?user=BUyRyUIAAAAJ&hl=en)

---

## Aim

This project develops a framework for estimating causal survival probabilities using reconstructed individual patient data (IPD) from published Kaplan–Meier curves, augmented with auxiliary summary information such as baseline characteristics reported in Table 1.

Within a single randomized trial, causal survival effects are identifiable under the original randomization scheme. However, when pooling reconstructed IPD across multiple studies, this identification no longer holds due to heterogeneity in baseline covariate distributions and study populations. In such settings, auxiliary paper-level summaries provide partial information about the underlying covariate structure, but do not fully determine it.

We propose a Bayesian nonparametric framework that operates at two levels. First, we model the unknown covariate distribution (or equivalently, reweighting scheme) using a flexible Dirichlet-based prior constrained by auxiliary summary statistics. Second, we explore nonparametric Bayesian survival modeling as an extension, allowing the underlying survival distribution itself to be modeled flexibly rather than relying solely on empirical estimators.

---

## Keywords

Bayesian nonparametrics, causal survival analysis, IPD reconstruction, Kaplan–Meier, Dirichlet process, summary statistics, transportability, reweighting, oncology trials, Dirichlet Process Mixture, Survival Models

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

### 2. Bayesian Nonparametric Reweighting (Design-Based)

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

### 3. Bayesian Nonparametric Survival Modeling (Extension)

In addition to design-based reweighting, we explore a model-based Bayesian nonparametric approach to survival estimation.

Rather than relying on empirical Kaplan–Meier estimators, we model the survival distribution using a Dirichlet process mixture model:

$$
G \sim \text{DP}(\alpha, G_0), \quad T_i \sim G,
$$

where $G$ represents an unknown survival distribution. In practice, this is implemented via a mixture model (e.g., mixtures of exponential or Weibull distributions), allowing flexible modeling of hazard heterogeneity.

This provides an alternative estimate of treatment-specific survival:

$$
S^a(t_0) = \mathbb{E}_G[\mathbf{1}(T > t_0) \mid A = a],
$$

which can be compared to the design-based estimator obtained via reweighting.

This extension serves two purposes:
- to assess sensitivity of results to the choice of survival model
- to connect the causal reweighting framework with fully generative Bayesian nonparametric modeling

### 4. Estimation of Causal Survival

Given weights $w$, treatment-specific survival probabilities are estimated as:

$$
S^a(t_0) = \sum_{i: A_i = a} w_i \cdot \mathbf{1}(T_i > t_0).
$$

Posterior inference integrates over the distribution of weights, yielding:
- posterior mean estimates of $S^a(t_0)$
- credible intervals reflecting uncertainty in covariate structure


### 5. Simulation Study

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

### 6. Extensions and Ongoing Work

This project naturally separates into two complementary directions:

**Causal inference with reconstructed IPD (current focus):**
- Robust estimation under covariate shift using auxiliary summaries
- Target-population weighting and uncertainty quantification

**Bayesian nonparametric modeling (ongoing extension):**
- Dirichlet process mixture models for survival distributions
- Comparison between design-based and model-based estimators
- Potential extension to dependent Dirichlet processes for study-specific heterogeneity

While the current implementation integrates both perspectives, future work may develop these directions independently as separate research projects.

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

### References

Lang, L., et al. (2024). RESOLVE-IPD: A high-fidelity approach to reconstructing individual patient data from Kaplan–Meier curves with subgroup information. [Link](https://arxiv.org/pdf/2511.01785)

Ying, S., et al. (2025). Summary-statistics-based causal inference under covariate shift: methods and applications [Link](https://arxiv.org/pdf/2603.02474v1)

Rott, K. W., et al. (2025). Causally interpretable meta-analysis combining aggregate and individual participant data. American Journal of Epidemiology, 194(7), 2060–2068. [Link](https://academic.oup.com/aje/article/194/7/2060/7762598)

Yao Zhao, Haoyue Sun, Yantian Ding, and Yanxun Xu. Km-gpt: An automated pipeline for reconstructing individual patient data from kaplan-meier plots, 2025. [Link](https://arxiv.org/abs/2509.18141)

### Datasets

Doi, T., Bennouna, J., Shen, L., Enzinger, P. C., Wang, R., Csiki, I., et al. (2016). KEYNOTE-181: Phase 3, open-label study of second-line pembrolizumab vs single-agent chemotherapy in patients with advanced/metastatic esophageal adenocarcinoma. Journal of Clinical Oncology, 34(15_suppl), TPS4140. [Link](https://ascopubs.org/doi/10.1200/JCO.2016.34.15_suppl.TPS4140)

Huang, J., Xu, J., Chen, Y., Zhuang, W., Zhang, Y., Chen, Z., et al. (2020). Camrelizumab versus investigator’s choice of chemotherapy as second-line therapy for advanced or metastatic oesophageal squamous cell carcinoma (ESCORT): A multicentre, randomised, open-label, phase 3 study. The Lancet Oncology, 21(6), 832–842 [Link](https://pubmed.ncbi.nlm.nih.gov/32416073/)

Kato, K., Cho, B. C., Takahashi, M., Okada, M., Lin, C. Y., Chin, K., et al. (2019). Nivolumab versus chemotherapy in patients with advanced oesophageal squamous cell carcinoma refractory or intolerant to previous chemotherapy (ATTRACTION-3): A multicentre, randomised, open-label, phase 3 trial. The Lancet Oncology, 20(11), 1506–1517 [Link](https://pubmed.ncbi.nlm.nih.gov/31582355/)

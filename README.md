Bayesian Nonparametric Survival Modeling of Cancer Outcomes Using Dirichlet Process Mixtures

This repository hosts the project for Bayesian nonparametric survival analysis of cancer outcomes using Dirichlet Process mixture models. The goal of this project is to explore flexible survival modeling approaches that relax strong parametric assumptions commonly used in classical survival analysis.

Traditional survival models such as exponential, Weibull, or Cox proportional hazards models assume specific functional forms for the hazard or survival distribution. In contrast, Dirichlet Process (DP) mixture models allow the survival distribution to be estimated nonparametrically, enabling the model to capture heterogeneous patient populations and potentially reveal latent survival subgroups.

Authors: J. Ma, S. Zhu

⸻

Aim

The primary objective of this project is to apply Bayesian nonparametric methods to cancer survival data in order to:
	•	flexibly estimate survival time distributions
	•	identify latent patient subgroups with distinct survival profiles
	•	compare nonparametric Bayesian models with classical survival models

⸻

Keywords

Bayesian nonparametrics, Dirichlet Process mixtures, survival analysis, cancer outcomes, biomedical statistics

⸻

Data Source

Cancer survival data will be obtained from publicly available biomedical datasets. Potential sources include:
	•	The Cancer Genome Atlas (TCGA) via the Genomic Data Commons (GDC)
	•	cBioPortal clinical datasets
	•	METABRIC breast cancer survival dataset

These datasets typically include variables such as:
	•	survival time
	•	censoring indicator (event vs. censored)
	•	demographic variables
	•	clinical covariates (e.g., tumor stage or treatment information)

⸻

Methodology

The central modeling framework is a Dirichlet Process mixture model for survival times. Rather than assuming a fixed parametric form for the survival distribution, the DP mixture model represents the distribution as a potentially infinite mixture of component distributions.

This approach allows the model to:
	•	flexibly estimate survival distributions
	•	capture multimodal survival behavior
	•	identify latent subgroups with different survival risks

Posterior inference will be performed using sampling-based methods, and results will be compared with classical survival models such as Kaplan–Meier estimators and parametric survival models.

⸻

Structure

```text
repo/
├── code/                        
│   └── requirements.txt          # Python dependencies
│
├── data/                         # Raw and processed datasets
│
├── docs/                         
│   ├── CHANGELOG.md              # Project updates and version history
│   ├── R_git.md                  # Quick reference for GitHub usage in R
│   └── Python_git.md             # Quick reference for GitHub usage in Python
│
├── notebooks/                    # Analysis notebooks for each project stage
│   ├── 01_data_acquisition.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_exploratory_survival_analysis.ipynb
│   ├── 04_parametric_survival_models.ipynb
│   ├── 05_dp_survival_model.ipynb
│   ├── 06_posterior_inference.ipynb
│   ├── 07_model_comparison.ipynb
│   └── 08_results_and_discussion.ipynb
│
├── .gitignore                    # Files and folders excluded from Git tracking
├── LICENSE                       # Usage license
└── README.md                     # Project overview
```

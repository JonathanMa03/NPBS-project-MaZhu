# Added
    - JM 3/20: Initial Repository Creation for BNP Survival Analysis
    - JM 3/21: Added API data aggregation and implementation of baseline survival model
    - JM 3/22: Added survival model and added BNP sampler
    - JM 3/24: added survival plots and wrote an implementation debrief notebook
    - JM 4/2:  Added RESOLVE-IPD as a submodule and tested reconstruction
    - JM 4/3:  Scanned through the papers used in KM-GPT for auxiliary summaries and added
    - JM 4/4:  Added a simulation study to test covariate augmentation
    - JM 4/5:  stress testing for covariate augmentation by definitions
    - JM 4/6:  Implemented Bayesian Bootstrapping and DP weighting 
    - JM 4/7:  Applied these two to both KM and RESOLVE to check for robustness
    - JM 4/8:  Added plots and experimental logs to a results and conclusion notebook
    - JM 4/14: Added a DP survival model so comparisone between pseudo-bayes and bayes can be done

# Changed
    - JM 3/22: Changed DeepSurv to a manually coded version since torch errors out
    - JM 4/3:  Changed synthetic reconstruction to outputs from KM-GPT and RESOLVE-IPD
    - JM 4/5:  Edited pipeline to accomodate causal estimands
    - JM 4/7:  Changed notebook to include markdown interpretation cells
    - JM 4/14: Updated implementation notebook and README to include DPSM


# Deleted
    - JM 4/1:  Deleted existing pipeline to shift from BNP Survival to using RESOLVE-IPD

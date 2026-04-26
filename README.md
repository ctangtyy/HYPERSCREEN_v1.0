# Pediatric Critical Care Hyperinflammatory Subphenotype Classifier

This repository contains R scripts for developing, validating, and visualizing a pediatric critical care classifier that predicts inflammatory subphenotype membership from routinely collected clinical variables. The binary outcome variable is expected to be `lca`, where `1 = hyperinflammatory` and `0 = hypoinflammatory`, and `study` identifies the contributing cohort.

> **Data note:** Patient-level clinical datasets are not included in this repository. To run the scripts, place the prepared analysis CSV at `datasheets/_all_subsetted_data.csv`, or update the paths in the run scripts.

## Project overview

The workflow has two related modeling pipelines:

1. **Step A: held-out cohort validation model**  
   The main classifier trains on derivation cohorts and evaluates on external validation cohorts. It performs median imputation, backward stepwise feature selection, elastic-net penalized logistic regression, AUROC/AUPRC evaluation with bootstrap confidence intervals, threshold-based metrics, ROC/PR plotting, confusion matrix plotting, and calibration plotting.

2. **Step B: all-data cross-validated model**  
   The Step B pipeline uses all cohorts together, adds one-hot encoded study indicators as predictors, performs stepwise feature selection, fits an elastic-net model, and evaluates out-of-fold cross-validated predictions.

Comparator models are also included for validation benchmarking: unpenalized logistic regression, SVM, random forest, and XGBoost.

## Repository structure

```text
.
├── README.md
├── .gitignore
├── package_versions.R
├── calibration_helpers.R
├── classifier_core.R
├── classifier_plots.R
├── classifier_stepB_core.R
├── classifier_stepB_plots.R
├── classifier_vs_comparators_plots.R
├── comparator_models.R
├── comparator_plots.R
├── run_models.R
├── run_stepB_model.R
├── datasheets/
│   └── _all_subsetted_data.csv        # not committed; prepared analysis data
└── results/                           # generated outputs; not committed by default
```

## Input data requirements

The primary input is a prepared CSV containing one row per patient/encounter and the following required columns:

| Column | Description |
|---|---|
| `lca` | Binary label, coded `1 = hyperinflammatory`, `0 = hypoinflammatory` |
| `study` | Cohort/study label, such as `cafpint`, `pali`, `redvent`, `cpccrn`, or `chop` |
| `id` | Optional patient/record identifier; excluded from predictors |
| `idstudy` | Optional identifier; excluded from predictors |

All candidate predictor columns must be numeric. The scripts median-impute missing predictor values before modeling.

## Main scripts

| Script | Purpose |
|---|---|
| `run_models.R` | Runs the held-out validation classifier and comparator models, then saves metrics, plots, coefficients, confusion matrices, and calibration plots. |
| `run_stepB_model.R` | Runs the Step B all-data cross-validated model and saves CV predictions, metrics, coefficients, plots, confusion matrices, and calibration plots. |
| `classifier_core.R` | Core Step A modeling functions: data loading, imputation, stepwise selection, elastic-net fitting, prediction, AUROC/AUPRC, bootstrap CIs, and threshold metrics. |
| `classifier_plots.R` | ROC, PR, and confusion matrix plotting for the held-out validation classifier. |
| `classifier_stepB_core.R` | Core Step B modeling functions, including study dummy variables and out-of-fold CV predictions. |
| `classifier_stepB_plots.R` | ROC, PR, and confusion matrix plotting for Step B CV predictions. |
| `comparator_models.R` | Fits GLM, SVM, random forest, and XGBoost comparator models. |
| `comparator_plots.R` | ROC, PR, and confusion matrix plotting for comparator models. |
| `classifier_vs_comparators_plots.R` | Overlays classifier and comparator ROC/PR curves. |
| `calibration_helpers.R` | Shared helper functions for binned calibration curves with exact binomial confidence intervals. |
| `package_versions.R` | Records package versions and session information for reproducibility. |

## Installation

Install the required packages from CRAN:

```r
install.packages(c(
  "data.table",
  "dplyr",
  "glmnet",
  "pROC",
  "PRROC",
  "olsrr",
  "ggplot2",
  "scales",
  "e1071",
  "randomForest",
  "xgboost",
  "tidyr"
))
```

To record the exact package versions used on your machine, run:

```r
source("package_versions.R")
```

This writes:

```text
package_versions.csv
sessionInfo.txt
```

Commit both files so the GitHub repository documents the R version and package versions used for the analysis.

Optional but recommended: use `renv` for a fully reproducible package environment.

```r
install.packages("renv")
renv::init()
renv::snapshot()
```

This creates an `renv.lock` file that can be committed. Future users can then restore the same package environment with:

```r
renv::restore()
```

## Running the analysis

First, place the prepared analysis CSV at:

```text
datasheets/_all_subsetted_data.csv
```

Then run the main held-out validation workflow:

```r
source("run_models.R")
```

Run the Step B all-data cross-validated workflow:

```r
source("run_stepB_model.R")
```

## Default model settings

The current run scripts use:

| Parameter | Value |
|---|---:|
| Elastic-net alpha | `0.15` |
| Cross-validation folds | `10` |
| Positive-class weight | `5.5` |
| Classification threshold | `0.7` |
| Bootstrap iterations | `1000` |
| Random seed | `0` |

## Generated outputs

The scripts write outputs under `results/`, including:

- classifier metrics
- comparator metrics
- selected predictors
- stepwise coefficients
- elastic-net coefficients
- ROC curves
- precision-recall curves
- classifier-vs-comparator plots
- confusion matrices
- calibration curves
- Step B cross-validated predictions

These generated outputs are ignored by default in `.gitignore`, except for optional summary files you may choose to commit manually.

## Reproducibility notes

- The modeling scripts set random seeds for reproducibility where applicable.
- Predictor variables are required to be numeric before modeling.
- Missing predictor values are median-imputed within the prepared CSV.
- AUROC and AUPRC confidence intervals are estimated using bootstrap resampling.
- External validation in Step A uses the default validation cohorts `cpccrn` and `chop`.
- Step B uses out-of-fold cross-validated predictions rather than held-out cohort validation.

## Privacy and sharing

Do not commit patient-level data, raw clinical files, or any file containing protected health information. Recommended files to keep out of GitHub include:

```text
datasheets/
results/
*.csv
*.tsv
*.xlsx
*.rds
*.RData
```

If you want to share example data, create a small fully synthetic CSV with the same column names and no real patient information.

## Citation / acknowledgement

This code was developed for research on pediatric critical care inflammatory subphenotype classification using routinely collected clinical variables. If reused, please cite the associated manuscript or project once available.

############################################################
# run_stepB_model.R
#
# Purpose:
#   Run Step B twice:
#     1) WITH redvent
#     2) WITHOUT redvent
#
# Step B:
#   - uses all rows in the CSV
#   - includes study dummy variables as predictors
#   - runs stepwise feature selection
#   - fits elastic net with CV predictions
#   - saves metrics, predictions, coefficients, and plots
############################################################

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
})

source("calibration_helpers.R")
source("classifier_stepB_core.R")
source("classifier_stepB_plots.R")

############################################################
# Helper: export glmnet coefficients
############################################################

save_stepB_coefficients <- function(model, file_path) {
  coef_sparse <- coef(model, s = "lambda.min")
  
  coef_df <- data.frame(
    predictor = rownames(coef_sparse),
    coefficient = as.numeric(coef_sparse),
    row.names = NULL
  )
  
  coef_df <- dplyr::filter(coef_df, coefficient != 0)
  
  data.table::fwrite(coef_df, file_path)
}

############################################################
# 1) Define run configurations
############################################################

run_configs <- list(
  list(
    label = "with_redvent",
    data_path = "datasheets/_all_subsetted_data.csv",
    results_dir = "results/stepB/with_redvent",
    plots_dir = "results/stepB/with_redvent/plots",
    confmat_dir = "results/stepB/with_redvent/confusion_matrices",
    calibration_dir = "results/stepB/with_redvent/calibration"
  ),
  list(
    label = "no_redvent",
    data_path = "datasheets/_all_subsetted_no_redvent.csv",
    results_dir = "results/stepB/no_redvent",
    plots_dir = "results/stepB/no_redvent/plots",
    confmat_dir = "results/stepB/no_redvent/confusion_matrices",
    calibration_dir = "results/stepB/no_redvent/calibration"
  )
)

############################################################
# 2) Shared parameters
############################################################

alpha_val <- 0.15
nfolds_val <- 10
class_weight_val <- 5.5
threshold_val <- 0.7
n_boot_val <- 1000
seed_val <- 0

############################################################
# 3) Run Step B for each configuration
############################################################

all_results <- list()

for (cfg in run_configs) {
  
  message("\n==============================")
  message("Running Step B pipeline: ", cfg$label)
  message("==============================\n")
  
  dir.create(cfg$results_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(cfg$plots_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(cfg$confmat_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(cfg$calibration_dir, recursive = TRUE, showWarnings = FALSE)
  
  res_stepB <- run_stepB_one_csv(
    csv_path = cfg$data_path,
    alpha = alpha_val,
    nfolds = nfolds_val,
    class_weight = class_weight_val,
    threshold = threshold_val,
    n_boot = n_boot_val,
    seed = seed_val
  )
  
  ##########################################################
  # Save tabular outputs
  ##########################################################
  
  data.table::fwrite(
    res_stepB$metrics,
    file.path(cfg$results_dir, paste0("stepB_metrics_", cfg$label, ".csv"))
  )
  
  data.table::fwrite(
    res_stepB$predictions,
    file.path(cfg$results_dir, paste0("stepB_cv_predictions_", cfg$label, ".csv"))
  )
  
  data.table::fwrite(
    res_stepB$stepwise_coefficients,
    file.path(cfg$results_dir, paste0("stepB_stepwise_coefficients_", cfg$label, ".csv"))
  )
  
  writeLines(
    res_stepB$predictors,
    file.path(cfg$results_dir, paste0("stepB_selected_predictors_", cfg$label, ".txt"))
  )
  
  save_stepB_coefficients(
    model = res_stepB$model,
    file_path = file.path(
      cfg$results_dir,
      paste0("stepB_elastic_net_coefficients_", cfg$label, ".csv")
    )
  )
  
  ##########################################################
  # Save plots
  ##########################################################
  
  stepB_plots <- plot_stepB_from_predictions(
    pred_df = res_stepB$predictions,
    threshold = threshold_val,
    out_dir = cfg$plots_dir,
    confmat_dir = cfg$confmat_dir,
    prefix = cfg$label,
    save_plots = TRUE,
    save_confmat = TRUE
  )
  
  stepB_calibration_plots <- save_stepB_calibration_plots(
    pred_df = res_stepB$predictions,
    out_dir = cfg$calibration_dir,
    prefix = cfg$label,
    n_bins_all = 10,
    n_bins_per_cohort = 8
  )
  
  ##########################################################
  # Store in memory
  ##########################################################
  
  all_results[[cfg$label]] <- list(
    stepB = res_stepB,
    plots = stepB_plots,
    calibration_plots = stepB_calibration_plots
  )
}

############################################################
# 4) Optional combined summary CSV
############################################################

stepB_summary <- data.table::rbindlist(
  lapply(names(all_results), function(lbl) {
    out <- all_results[[lbl]]$stepB$metrics
    out$run_label <- lbl
    out
  }),
  fill = TRUE
)

data.table::fwrite(
  stepB_summary,
  "results/stepB/stepB_metrics_all_runs.csv"
)

message("\nAll Step B runs completed successfully.\n")


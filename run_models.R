############################################################
# run_models.R
#
# Purpose:
#   Run the classifier pipeline and comparator models for:
#     1) WITH redvent
#     2) WITHOUT redvent
#   and save outputs into separate result folders.
############################################################

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
})

############################################################
# 0) Load all pipeline code
############################################################

source("classifier_core.R")
source("calibration_helpers.R")
source("classifier_plots.R")
source("comparator_models.R")
source("comparator_plots.R")
source("classifier_vs_comparators_plots.R")

############################################################
# Helper: export glmnet coefficients
############################################################

save_classifier_coefficients <- function(model, file_path) {
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
    derivation_studies = c("cafpint", "pali", "redvent"),
    validation_studies = c("cpccrn", "chop"),
    results_dir = "results/with_redvent",
    plots_dir = "results/with_redvent/plots",
    confmat_dir = "results/with_redvent/confusion_matrices",
    calibration_dir = "results/with_redvent/calibration"
  ),
  list(
    label = "no_redvent",
    data_path = "datasheets/_all_subsetted_no_redvent.csv",
    derivation_studies = c("cafpint", "pali"),
    validation_studies = c("cpccrn", "chop"),
    results_dir = "results/no_redvent",
    plots_dir = "results/no_redvent/plots",
    confmat_dir = "results/no_redvent/confusion_matrices",
    calibration_dir = "results/no_redvent/calibration"
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
# 3) Run everything for each configuration
############################################################

all_results <- list()

for (cfg in run_configs) {
  
  message("\n==============================")
  message("Running pipeline: ", cfg$label)
  message("==============================\n")
  
  dir.create(cfg$results_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(cfg$plots_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(cfg$confmat_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(cfg$calibration_dir, recursive = TRUE, showWarnings = FALSE)
  
  ##########################################################
  # 3A) Run + save classifier
  ##########################################################
  
  res <- run_one_csv(
    csv_path = cfg$data_path,
    derivation_studies = cfg$derivation_studies,
    validation_studies = cfg$validation_studies,
    alpha = alpha_val,
    nfolds = nfolds_val,
    class_weight = class_weight_val,
    threshold = threshold_val,
    n_boot = n_boot_val,
    seed = seed_val
  )
  
  data.table::fwrite(
    res$metrics,
    file.path(cfg$results_dir, paste0("classifier_metrics_", cfg$label, ".csv"))
  )
  
  save_classifier_coefficients(
    model = res$model,
    file_path = file.path(
      cfg$results_dir,
      paste0("classifier_coefficients_", cfg$label, ".csv")
    )
  )
  
  writeLines(
    res$predictors,
    con = file.path(cfg$results_dir, paste0("stepwise_selected_predictors_", cfg$label, ".txt"))
  )
  
  data.table::fwrite(
    res$stepwise_coefficients,
    file.path(cfg$results_dir, paste0("stepwise_coefficients_", cfg$label, ".csv"))
  )
  
  print(res$predictors)
  
  plots <- plot_from_trained_model(
    csv_path = cfg$data_path,
    model = res$model,
    classifier_predictors = res$predictors,
    validation_studies = cfg$validation_studies,
    out_dir = cfg$plots_dir,
    prefix = cfg$label,
    threshold = threshold_val,
    save_confmats = TRUE,
    confmat_dir = cfg$confmat_dir
  )
  
  stepA_calibration_plots <- save_stepA_calibration_plots(
    pred_df = plots$pred_df,
    out_dir = cfg$calibration_dir,
    prefix = cfg$label,
    n_bins_all = 10,
    n_bins_per_cohort = 8
  )
  
  ##########################################################
  # 3B) Run + save comparator models
  ##########################################################
  
  res_cmp <- run_comparator_models_one_csv(
    csv_path = cfg$data_path,
    derivation_studies = cfg$derivation_studies,
    validation_studies = cfg$validation_studies,
    threshold = threshold_val,
    class_weight = class_weight_val,
    n_boot = n_boot_val,
    seed = seed_val
  )
  
  data.table::fwrite(
    res_cmp$metrics,
    file.path(cfg$results_dir, paste0("comparator_metrics_", cfg$label, ".csv"))
  )
  
  roc_p <- plot_roc_compare(res_cmp$preds)
  pr_p  <- plot_pr_compare(res_cmp$preds)
  
  save_comparator_plots(
    roc_plot = roc_p,
    pr_plot = pr_p,
    out_dir = cfg$plots_dir,
    prefix = cfg$label
  )
  
  cm_table <- print_confusion_matrices(
    res_cmp$preds,
    threshold = threshold_val
  )
  
  data.table::fwrite(
    cm_table,
    file.path(cfg$results_dir, paste0("comparator_confusion_metrics_", cfg$label, ".csv"))
  )
  
  save_confusion_matrices_png(
    preds_df = res_cmp$preds,
    threshold = threshold_val,
    out_dir = cfg$confmat_dir,
    prefix = cfg$label
  )
  
  ##########################################################
  # 3C) Save combined classifier-vs-comparators plots
  ##########################################################
  
  combo_plots <- plot_classifier_vs_comparators(
    csv_path = cfg$data_path,
    classifier_model = res$model,
    classifier_predictors = res$predictors,
    comparator_preds_df = res_cmp$preds,
    validation_studies = cfg$validation_studies,
    classifier_label = "Classifier",
    out_dir = cfg$plots_dir,
    prefix = cfg$label,
    save_plots = TRUE
  )
  
  ##########################################################
  # 3D) Store in memory if you want to inspect later
  ##########################################################
  
  all_results[[cfg$label]] <- list(
    classifier = res,
    classifier_plots = plots,
    comparators = res_cmp,
    comparator_roc = roc_p,
    comparator_pr = pr_p,
    comparator_confusion_table = cm_table,
    combo_plots = combo_plots,
    stepA_calibration_plots = stepA_calibration_plots
  )
}

############################################################
# 4) Optional combined summary CSVs
############################################################

classifier_summary <- data.table::rbindlist(
  lapply(names(all_results), function(lbl) {
    out <- all_results[[lbl]]$classifier$metrics
    out$run_label <- lbl
    out
  }),
  fill = TRUE
)

comparator_summary <- data.table::rbindlist(
  lapply(names(all_results), function(lbl) {
    out <- all_results[[lbl]]$comparators$metrics
    out$run_label <- lbl
    out
  }),
  fill = TRUE
)

data.table::fwrite(
  classifier_summary,
  "results/classifier_metrics_all_runs.csv"
)

data.table::fwrite(
  comparator_summary,
  "results/comparator_metrics_all_runs.csv"
)

message("\nAll runs completed successfully.\n")

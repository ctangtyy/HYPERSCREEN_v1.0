############################################################
# classifier_core.R
#
# Purpose:
#   Core functions to:
#     1) Load one analysis CSV (already assembled upstream)
#     2) Median-impute predictors across the entire CSV
#     3) Perform backward stepwise logistic regression feature selection
#        using p-value-based elimination (olsrr)
#     4) Fit elastic-net penalized logistic regression on derivation cohorts
#     5) Evaluate AUROC + AUPRC (with bootstrap CIs) overall + per cohort
#     6) Compute threshold-based classification metrics at p > 0.7
#
# Key conventions expected in the input CSV:
#   - Column 'lca' is the binary outcome (0/1): 1 = hyper, 0 = hypo
#   - Column 'study' identifies cohort (e.g., cafpint, pali, redvent, cpccrn, chop)
#   - Optional columns 'id' and 'idstudy' are identifiers and are NOT predictors
#
# Notes:
#   - This script is the held-out cohort validation pipeline (poster Script A),
#     not the all-data LOOCV + cohort-variable pipeline (poster Script B).
#   - It is written to work with both:
#       * _all_subsetted_data.csv
#       * _all_subsetted_no_redvent.csv
#
############################################################

suppressPackageStartupMessages({
  library(data.table)  # fread
  library(dplyr)       # data wrangling
  library(glmnet)      # elastic net
  library(pROC)        # AUROC
  library(PRROC)       # AUPRC
  library(olsrr)       # backward stepwise selection
})

############################################################
# 0) Small utilities
############################################################

#' Check that required columns exist.
assert_required_cols <- function(df, required = c("lca", "study")) {
  missing <- setdiff(required, colnames(df))
  if (length(missing) > 0) {
    stop(sprintf(
      "Input data is missing required columns: %s",
      paste(missing, collapse = ", ")
    ))
  }
}

#' Identify candidate predictor columns.
#' Excludes identifiers, cohort labels, and outcome.
get_predictor_names <- function(df) {
  preds <- df %>%
    dplyr::select(-any_of(c("id", "study", "lca", "idstudy"))) %>%
    colnames()
  
  if (length(preds) == 0) {
    stop("No predictors found after excluding id/study/lca/idstudy.")
  }
  
  preds
}

#' Median imputation across the ENTIRE dataset (within the CSV).
#' glmnet cannot handle NA values.
median_impute_all <- function(df, predictor_names) {
  out <- df
  
  for (p in predictor_names) {
    if (!is.numeric(out[[p]])) {
      stop(sprintf(
        "Predictor '%s' is not numeric. glmnet requires numeric predictors.",
        p
      ))
    }
    
    med <- median(out[[p]], na.rm = TRUE)
    
    if (is.na(med)) {
      stop(sprintf(
        "Predictor '%s' appears to be all NA. Drop it upstream or fix the input CSV.",
        p
      ))
    }
    
    out[[p]][is.na(out[[p]])] <- med
  }
  
  out
}

#' Perform backward stepwise logistic regression feature selection.
#'
#' Mirrors the poster methods wording:
#'   - fit logistic regression with all candidate predictors
#'   - run backward stepwise selection by p-value (olsrr)
#'   - return selected predictor names
#'
#' This is done on the imputed CSV, consistent with the original workflow style.
select_predictors_stepwise <- function(
    df_imputed,
    candidate_predictors,
    p_enter = 0.05,
    p_remove = 0.05
) {
  if (length(candidate_predictors) == 0) {
    stop("No candidate predictors supplied to stepwise selection.")
  }
  
  step_df <- df_imputed %>%
    dplyr::select(all_of(c("lca", candidate_predictors)))
  
  # Full logistic regression with all candidate predictors
  full_mod <- glm(lca ~ ., data = step_df, family = "binomial")
  
  # Backward stepwise by p-value
  step_mod <- olsrr::ols_step_backward_p(
    full_mod,
    prem = p_remove,
    pent = p_enter,
    details = FALSE
  )
  
  selected <- names(step_mod$model$coefficients)
  selected <- setdiff(selected, "(Intercept)")
  
  if (length(selected) == 0) {
    stop("Backward stepwise selection returned no predictors.")
  }
  
  list(
    selected_predictors = selected,
    stepwise_model = step_mod,
    full_model = full_mod
  )
}

#' Helper: extract stepwise logistic regression coefficients
extract_stepwise_coefficients <- function(step_model) {
  if (is.null(step_model$model)) {
    stop("step_model$model is missing; cannot extract stepwise coefficients.")
  }
  
  coef_vec <- coef(step_model$model)
  
  data.frame(
    predictor = names(coef_vec),
    coefficient = as.numeric(coef_vec),
    row.names = NULL
  )
}

############################################################
# 1) Data loading
############################################################

#' Load one CSV produced by the upstream pipeline.
#' @param csv_path file path to a dataset CSV
#' @return dataframe with verified required columns
load_dataset_csv <- function(csv_path) {
  df <- data.frame(fread(csv_path))
  assert_required_cols(df, c("lca", "study"))
  
  # Ensure outcome is numeric 0/1
  if (is.factor(df$lca)) df$lca <- as.character(df$lca)
  df$lca <- as.numeric(df$lca)
  
  if (!all(df$lca %in% c(0, 1))) {
    stop("Column 'lca' must be coded as 0/1.")
  }
  
  df$study <- as.character(df$study)
  df
}

############################################################
# 2) Model fitting (elastic net logistic regression)
############################################################

#' Fit an elastic-net penalized logistic regression using cv.glmnet.
#'
#' This is the held-out cohort validation model, so 10-fold CV is used
#' here to tune lambda within the derivation cohorts.
fit_elastic_net <- function(
    df_imputed,
    predictor_names,
    derivation_studies = c("cafpint", "pali", "redvent"),
    alpha = 0.15,
    nfolds = 10,
    class_weight = 5.5,
    seed = 0
) {
  set.seed(seed)
  
  train_df <- df_imputed %>% filter(study %in% derivation_studies)
  
  if (nrow(train_df) == 0) {
    stop("No rows found for derivation_studies in this CSV.")
  }
  
  x_train <- train_df %>%
    dplyr::select(all_of(predictor_names)) %>%
    as.matrix()
  
  y_train <- train_df$lca
  
  # Positive-class weighting
  w <- rep(1, length(y_train))
  w[y_train == 1] <- class_weight
  
  cvfit <- cv.glmnet(
    x = x_train,
    y = y_train,
    family = "binomial",
    alpha = alpha,
    weights = w,
    type.measure = "deviance",
    nfolds = nfolds
  )
  
  list(
    cvfit = cvfit,
    predictor_names = predictor_names,
    derivation_studies = derivation_studies
  )
}

############################################################
# 3) Prediction + metrics
############################################################

#' Predict probabilities for class 1 (hyper).
predict_prob <- function(cvfit, x) {
  as.numeric(predict(cvfit, newx = x, s = "lambda.min", type = "response"))
}

#' Compute AUROC from true labels and predicted probabilities.
compute_auroc <- function(y_true, p_hat) {
  roc_obj <- pROC::roc(
    response = y_true,
    predictor = p_hat,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )
  as.numeric(pROC::auc(roc_obj))
}

#' Compute AUPRC from true labels and predicted probabilities.
compute_auprc <- function(y_true, p_hat) {
  pr <- PRROC::pr.curve(
    scores.class0 = p_hat[y_true == 1],
    scores.class1 = p_hat[y_true == 0],
    curve = FALSE
  )
  as.numeric(pr$auc.integral)
}

#' Threshold-based confusion metrics at a fixed probability cutoff.
compute_threshold_metrics <- function(y_true, p_hat, threshold = 0.7) {
  y_pred <- ifelse(p_hat > threshold, 1, 0)
  
  tp <- sum(y_pred == 1 & y_true == 1)
  fp <- sum(y_pred == 1 & y_true == 0)
  tn <- sum(y_pred == 0 & y_true == 0)
  fn <- sum(y_pred == 0 & y_true == 1)
  
  sens <- ifelse((tp + fn) == 0, NA_real_, tp / (tp + fn))
  spec <- ifelse((tn + fp) == 0, NA_real_, tn / (tn + fp))
  ppv  <- ifelse((tp + fp) == 0, NA_real_, tp / (tp + fp))
  npv  <- ifelse((tn + fn) == 0, NA_real_, tn / (tn + fn))
  
  list(
    threshold = threshold,
    tp = tp, fp = fp, tn = tn, fn = fn,
    sensitivity = sens,
    specificity = spec,
    ppv = ppv,
    npv = npv
  )
}

############################################################
# 4) Bootstrap CIs for AUROC/AUPRC
############################################################

#' Bootstrap CI for AUROC and AUPRC on a fixed evaluation set.
#' Resamples evaluation rows with replacement; does not refit the model.
bootstrap_metric_ci <- function(
    y_true,
    p_hat,
    metric_fun,
    n_boot = 1000,
    seed = 0
) {
  set.seed(seed)
  
  n <- length(y_true)
  vals <- numeric(n_boot)
  
  for (b in seq_len(n_boot)) {
    idx <- sample.int(n, size = n, replace = TRUE)
    vals[b] <- metric_fun(y_true[idx], p_hat[idx])
  }
  
  ci <- stats::quantile(vals, probs = c(0.025, 0.5, 0.975), na.rm = TRUE)
  
  list(
    boot_values = vals,
    median = as.numeric(ci[[2]]),
    ci_low = as.numeric(ci[[1]]),
    ci_high = as.numeric(ci[[3]])
  )
}

############################################################
# 5) Evaluation wrapper: overall + per cohort
############################################################

#' Evaluate a fitted model on validation cohorts.
evaluate_model <- function(
    df_imputed,
    fit_obj,
    validation_studies = c("cpccrn", "chop"),
    threshold = 0.7,
    n_boot = 1000,
    seed = 0
) {
  cvfit <- fit_obj$cvfit
  predictor_names <- fit_obj$predictor_names
  
  eval_df <- df_imputed %>% filter(study %in% validation_studies)
  if (nrow(eval_df) == 0) {
    stop("No rows found for validation_studies in this CSV.")
  }
  
  x_eval <- eval_df %>%
    dplyr::select(all_of(predictor_names)) %>%
    as.matrix()
  
  y_eval <- eval_df$lca
  p_eval <- predict_prob(cvfit, x_eval)
  
  # Overall metrics
  auroc <- compute_auroc(y_eval, p_eval)
  auprc <- compute_auprc(y_eval, p_eval)
  
  auroc_ci <- bootstrap_metric_ci(
    y_true = y_eval,
    p_hat = p_eval,
    metric_fun = compute_auroc,
    n_boot = n_boot,
    seed = seed
  )
  
  auprc_ci <- bootstrap_metric_ci(
    y_true = y_eval,
    p_hat = p_eval,
    metric_fun = compute_auprc,
    n_boot = n_boot,
    seed = seed + 1
  )
  
  thr <- compute_threshold_metrics(y_eval, p_eval, threshold = threshold)
  
  overall_row <- data.frame(
    cohort = "ALL_VALIDATION",
    n = length(y_eval),
    auroc = auroc,
    auroc_ci_low = auroc_ci$ci_low,
    auroc_ci_high = auroc_ci$ci_high,
    auprc = auprc,
    auprc_ci_low = auprc_ci$ci_low,
    auprc_ci_high = auprc_ci$ci_high,
    threshold = thr$threshold,
    sensitivity = thr$sensitivity,
    specificity = thr$specificity,
    ppv = thr$ppv,
    npv = thr$npv
  )
  
  # Per-cohort metrics
  per_cohort <- eval_df %>%
    dplyr::select(study, lca) %>%
    dplyr::mutate(prob = p_eval) %>%
    dplyr::group_by(study) %>%
    dplyr::group_modify(function(dat, key) {
      y <- dat$lca
      p <- dat$prob
      
      if (length(unique(y)) < 2) {
        return(data.frame(
          cohort = as.character(key$study),
          n = nrow(dat),
          auroc = NA, auroc_ci_low = NA, auroc_ci_high = NA,
          auprc = NA, auprc_ci_low = NA, auprc_ci_high = NA,
          threshold = threshold,
          sensitivity = NA, specificity = NA, ppv = NA, npv = NA
        ))
      }
      
      a1 <- compute_auroc(y, p)
      p1 <- compute_auprc(y, p)
      
      a1_ci <- bootstrap_metric_ci(
        y, p, compute_auroc,
        n_boot = n_boot,
        seed = seed
      )
      
      p1_ci <- bootstrap_metric_ci(
        y, p, compute_auprc,
        n_boot = n_boot,
        seed = seed + 1
      )
      
      t1 <- compute_threshold_metrics(y, p, threshold = threshold)
      
      data.frame(
        cohort = as.character(key$study),
        n = nrow(dat),
        auroc = a1,
        auroc_ci_low = a1_ci$ci_low,
        auroc_ci_high = a1_ci$ci_high,
        auprc = p1,
        auprc_ci_low = p1_ci$ci_low,
        auprc_ci_high = p1_ci$ci_high,
        threshold = t1$threshold,
        sensitivity = t1$sensitivity,
        specificity = t1$specificity,
        ppv = t1$ppv,
        npv = t1$npv
      )
    }) %>%
    dplyr::ungroup()
  
  dplyr::bind_rows(overall_row, per_cohort)
}

############################################################
# 6) One-call pipeline for a single CSV
############################################################

#' Run the full held-out validation pipeline on one CSV.
#'
#' Example:
#'   # With REDVENT in derivation
#'   res_with <- run_one_csv(
#'     csv_path = "_all_subsetted_data.csv",
#'     derivation_studies = c("cafpint", "pali", "redvent"),
#'     validation_studies = c("cpccrn", "chop")
#'   )
#'
#'   # Without REDVENT in derivation
#'   res_no <- run_one_csv(
#'     csv_path = "_all_subsetted_no_redvent.csv",
#'     derivation_studies = c("cafpint", "pali"),
#'     validation_studies = c("cpccrn", "chop")
#'   )
run_one_csv <- function(
    csv_path,
    derivation_studies = c("cafpint", "pali", "redvent"),
    validation_studies = c("cpccrn", "chop"),
    alpha = 0.15,
    nfolds = 10,
    class_weight = 5.5,
    threshold = 0.7,
    n_boot = 1000,
    seed = 0,
    stepwise_p_enter = 0.05,
    stepwise_p_remove = 0.05
) {
  # 1) Load raw data
  df <- load_dataset_csv(csv_path)
  
  # Sanity-check requested cohorts exist in this CSV
  missing_deriv <- setdiff(derivation_studies, unique(df$study))
  if (length(missing_deriv) > 0) {
    stop(sprintf(
      "These derivation_studies are not present in the CSV: %s",
      paste(missing_deriv, collapse = ", ")
    ))
  }
  
  missing_valid <- setdiff(validation_studies, unique(df$study))
  if (length(missing_valid) > 0) {
    stop(sprintf(
      "These validation_studies are not present in the CSV: %s",
      paste(missing_valid, collapse = ", ")
    ))
  }
  
  # 2) Determine all candidate predictors
  candidate_predictors <- get_predictor_names(df)
  
  # 3) Median impute candidate predictors across the full CSV
  df_imp <- median_impute_all(df, candidate_predictors)
  
  # 4) Backward stepwise logistic regression feature selection
  stepwise_res <- select_predictors_stepwise(
    df_imputed = df_imp,
    candidate_predictors = candidate_predictors,
    p_enter = stepwise_p_enter,
    p_remove = stepwise_p_remove
  )
  
  predictor_names <- stepwise_res$selected_predictors
  stepwise_coef_df <- extract_stepwise_coefficients(stepwise_res$stepwise_model)
  
  message("Stepwise-selected predictors (", length(predictor_names), "):")
  message(paste(predictor_names, collapse = ", "))
  
  # 5) Fit elastic net on derivation cohorts using selected predictors
  fit_obj <- fit_elastic_net(
    df_imputed = df_imp,
    predictor_names = predictor_names,
    derivation_studies = derivation_studies,
    alpha = alpha,
    nfolds = nfolds,
    class_weight = class_weight,
    seed = seed
  )
  
  # 6) Evaluate on validation cohorts
  metrics_df <- evaluate_model(
    df_imputed = df_imp,
    fit_obj = fit_obj,
    validation_studies = validation_studies,
    threshold = threshold,
    n_boot = n_boot,
    seed = seed
  )
  
  metrics_df$dataset <- basename(csv_path)
  
  list(
    metrics = metrics_df,
    model = fit_obj$cvfit,
    predictors = predictor_names,
    candidate_predictors = candidate_predictors,
    derivation_studies = derivation_studies,
    validation_studies = validation_studies,
    stepwise_model = stepwise_res$stepwise_model,
    full_glm_model = stepwise_res$full_model,
    stepwise_coefficients = stepwise_coef_df
  )
}
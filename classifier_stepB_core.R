############################################################
# classifier_stepB_core.R
#
# Purpose:
#   Step B model:
#     1) Load one full analysis CSV
#     2) Add study dummy variables as predictors
#     3) Median-impute predictors across the full CSV
#     4) Perform backward stepwise logistic regression feature selection
#     5) Fit elastic-net penalized logistic regression on ALL rows
#     6) Generate out-of-fold CV predictions using cv.glmnet(keep=TRUE)
#     7) Evaluate AUROC + AUPRC + threshold metrics on CV predictions
#
# Notes:
#   - Uses ALL cohorts together
#   - Includes study variables as predictors
#   - Uses CV predictions, not held-out cohort validation
############################################################

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(glmnet)
  library(pROC)
  library(PRROC)
  library(olsrr)
})

############################################################
# 0) Shared utilities
############################################################

assert_required_cols <- function(df, required = c("lca", "study")) {
  missing <- setdiff(required, colnames(df))
  if (length(missing) > 0) {
    stop(sprintf(
      "Input data is missing required columns: %s",
      paste(missing, collapse = ", ")
    ))
  }
}

load_dataset_csv <- function(csv_path) {
  df <- data.frame(data.table::fread(csv_path))
  assert_required_cols(df, c("lca", "study"))
  
  if (is.factor(df$lca)) df$lca <- as.character(df$lca)
  df$lca <- as.numeric(df$lca)
  
  if (!all(df$lca %in% c(0, 1))) {
    stop("Column 'lca' must be coded as 0/1.")
  }
  
  df$study <- as.character(df$study)
  df
}

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

compute_auprc <- function(y_true, p_hat) {
  pr <- PRROC::pr.curve(
    scores.class0 = p_hat[y_true == 1],
    scores.class1 = p_hat[y_true == 0],
    curve = FALSE
  )
  as.numeric(pr$auc.integral)
}

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
# 1) Step B-specific preprocessing
############################################################

# Add one-hot study indicators as predictors.
# Keep original study column too for reference / plotting.
add_study_dummies <- function(df, reference_study = "cafpint") {
  df$study <- factor(df$study)
  
  if (!reference_study %in% levels(df$study)) {
    stop(sprintf(
      "reference_study '%s' is not present in df$study.",
      reference_study
    ))
  }
  
  df$study <- stats::relevel(df$study, ref = reference_study)
  
  # model.matrix(~ study) gives:
  #   (Intercept) + k-1 dummy columns
  study_mat <- stats::model.matrix(~ study, data = df)[, -1, drop = FALSE]
  
  study_df <- as.data.frame(study_mat, check.names = FALSE)
  
  cbind(df, study_df)
}

# In Step B we exclude raw study, outcome, IDs,
# but KEEP the study dummy columns.
get_predictor_names_stepB <- function(df) {
  preds <- df %>%
    dplyr::select(-any_of(c("id", "idstudy", "study", "lca"))) %>%
    colnames()
  
  if (length(preds) == 0) {
    stop("No predictors found for Step B after excluding id/idstudy/study/lca.")
  }
  
  preds
}

############################################################
# 2) Stepwise feature selection
############################################################

select_predictors_stepwise_stepB <- function(
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
  
  full_mod <- glm(lca ~ ., data = step_df, family = "binomial")
  
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

############################################################
# 3) Fit elastic net on ALL rows with CV predictions
############################################################

fit_elastic_net_stepB <- function(
    df_imputed,
    predictor_names,
    alpha = 0.15,
    nfolds = 10,
    class_weight = 5.5,
    seed = 0
) {
  set.seed(seed)
  
  x <- df_imputed %>%
    dplyr::select(all_of(predictor_names)) %>%
    as.matrix()
  
  y <- df_imputed$lca
  
  w <- rep(1, length(y))
  w[y == 1] <- class_weight
  
  cvfit <- glmnet::cv.glmnet(
    x = x,
    y = y,
    family = "binomial",
    alpha = alpha,
    weights = w,
    type.measure = "deviance",
    nfolds = nfolds,
    keep = TRUE
  )
  
  list(
    cvfit = cvfit,
    predictor_names = predictor_names
  )
}

############################################################
# 4) Extract out-of-fold predictions at lambda.min
############################################################

get_cv_pred_probs <- function(cvfit) {
  if (is.null(cvfit$fit.preval)) {
    stop("cvfit$fit.preval is missing. Make sure cv.glmnet was run with keep = TRUE.")
  }
  
  lambda_idx <- which.min(abs(cvfit$lambda - cvfit$lambda.min))
  p_cv <- cvfit$fit.preval[, lambda_idx]
  
  p_cv <- as.numeric(p_cv)
  
  if (any(is.na(p_cv))) {
    stop("Out-of-fold CV predictions contain NA values.")
  }
  
  p_cv
}

############################################################
# 5) Evaluate Step B model using CV predictions
############################################################

evaluate_stepB_cv_predictions <- function(
    df_imputed,
    p_cv,
    threshold = 0.7,
    n_boot = 1000,
    seed = 0
) {
  y_true <- df_imputed$lca
  
  if (length(y_true) != length(p_cv)) {
    stop("Length mismatch between true labels and CV predictions.")
  }
  
  auroc <- compute_auroc(y_true, p_cv)
  auprc <- compute_auprc(y_true, p_cv)
  
  auroc_ci <- bootstrap_metric_ci(
    y_true = y_true,
    p_hat = p_cv,
    metric_fun = compute_auroc,
    n_boot = n_boot,
    seed = seed
  )
  
  auprc_ci <- bootstrap_metric_ci(
    y_true = y_true,
    p_hat = p_cv,
    metric_fun = compute_auprc,
    n_boot = n_boot,
    seed = seed + 1
  )
  
  thr <- compute_threshold_metrics(y_true, p_cv, threshold = threshold)
  
  metrics_df <- data.frame(
    cohort = "ALL_DATA_CV",
    n = length(y_true),
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
  
  pred_df <- df_imputed %>%
    dplyr::select(any_of(c("study", "id", "idstudy")), lca) %>%
    dplyr::mutate(prob = p_cv)
  
  list(
    metrics = metrics_df,
    predictions = pred_df
  )
}

############################################################
# 6) One-call Step B pipeline
############################################################

run_stepB_one_csv <- function(
    csv_path,
    alpha = 0.15,
    nfolds = 10,
    class_weight = 5.5,
    threshold = 0.7,
    n_boot = 1000,
    seed = 0,
    stepwise_p_enter = 0.05,
    stepwise_p_remove = 0.05
) {
  # 1) Load
  df <- load_dataset_csv(csv_path)
  
  # 2) Add study dummy variables
  df_aug <- add_study_dummies(df)
  
  # 3) Identify candidate predictors
  candidate_predictors <- get_predictor_names_stepB(df_aug)
  
  # 4) Median impute all predictors
  df_imp <- median_impute_all(df_aug, candidate_predictors)
  
  # 5) Stepwise feature selection
  stepwise_res <- select_predictors_stepwise_stepB(
    df_imputed = df_imp,
    candidate_predictors = candidate_predictors,
    p_enter = stepwise_p_enter,
    p_remove = stepwise_p_remove
  )
  
  predictor_names <- stepwise_res$selected_predictors
  stepwise_coef_df <- extract_stepwise_coefficients(stepwise_res$stepwise_model)
  
  message("Step B stepwise-selected predictors (", length(predictor_names), "):")
  message(paste(predictor_names, collapse = ", "))
  
  # 6) Elastic net on all rows with CV predictions
  fit_obj <- fit_elastic_net_stepB(
    df_imputed = df_imp,
    predictor_names = predictor_names,
    alpha = alpha,
    nfolds = nfolds,
    class_weight = class_weight,
    seed = seed
  )
  
  # 7) Get out-of-fold predictions
  p_cv <- get_cv_pred_probs(fit_obj$cvfit)
  
  # 8) Evaluate
  eval_res <- evaluate_stepB_cv_predictions(
    df_imputed = df_imp,
    p_cv = p_cv,
    threshold = threshold,
    n_boot = n_boot,
    seed = seed
  )
  
  eval_res$metrics$dataset <- basename(csv_path)
  
  list(
    metrics = eval_res$metrics,
    predictions = eval_res$predictions,
    cv_probabilities = p_cv,
    model = fit_obj$cvfit,
    predictors = predictor_names,
    candidate_predictors = candidate_predictors,
    stepwise_model = stepwise_res$stepwise_model,
    full_glm_model = stepwise_res$full_model,
    stepwise_coefficients = stepwise_coef_df
  )
}
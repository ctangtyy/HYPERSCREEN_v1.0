############################################################
# comparator_models.R
#
# Purpose:
#   Fit comparator models (GLM, SVM, RF, XGB) and evaluate
#   AUROC + AUPRC (bootstrap CIs) overall + per cohort,
#   using the same conventions as classifier_core.R.
#
# Dependencies:
#   - This file SOURCES classifier_core.R for shared utilities:
#       load_dataset_csv()
#       get_predictor_names()
#       median_impute_all()
#       select_predictors_stepwise()
#       compute_auroc()
#       compute_auprc()
#       compute_threshold_metrics()
#       bootstrap_metric_ci()
#
############################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(data.table)
  library(pROC)
  library(PRROC)
  
  library(e1071)        # SVM
  library(randomForest) # RF
  library(xgboost)      # XGB
  library(ggplot2)
})

source("classifier_core.R")

############################################################
# 1) Train/predict wrappers (each returns prob for class 1)
############################################################

# --- (A) Plain logistic regression (no penalty) ---
fit_predict_glm <- function(train_df, valid_df, predictor_names, class_weight = 1) {
  w <- rep(1, nrow(train_df))
  w[train_df$lca == 1] <- as.integer(round(class_weight))
  
  f <- as.formula(paste("lca ~", paste(predictor_names, collapse = " + ")))
  mod <- glm(f, data = train_df, family = binomial(), weights = w)
  
  p_valid <- as.numeric(predict(mod, newdata = valid_df, type = "response"))
  list(model = mod, prob_valid = p_valid)
}

# --- (B) SVM (probability output) ---
fit_predict_svm <- function(train_df, valid_df, predictor_names,
                            class_weight = 1, seed = 0) {
  set.seed(seed)
  
  x_train <- train_df %>% dplyr::select(all_of(predictor_names)) %>% as.matrix()
  y_train <- factor(as.integer(train_df$lca), levels = c(0, 1))
  
  x_valid <- valid_df %>% dplyr::select(all_of(predictor_names)) %>% as.matrix()
  
  default_gamma <- 1 / ncol(x_train)
  
  svm_fit <- e1071::svm(
    x = x_train,
    y = y_train,
    kernel = "radial",
    gamma = default_gamma,
    cost = 1,
    probability = TRUE,
    class.weights = c("0" = 1, "1" = class_weight)
  )
  
  pred <- predict(svm_fit, x_valid, probability = TRUE)
  prob_attr <- attr(pred, "probabilities")
  if (is.null(prob_attr)) stop("SVM did not return probabilities; check probability=TRUE and factor y.")
  
  if (!("1" %in% colnames(prob_attr))) {
    stop(paste0(
      "SVM probability matrix missing class '1' column. Columns: ",
      paste(colnames(prob_attr), collapse = ", ")
    ))
  }
  
  p_valid <- as.numeric(prob_attr[, "1"])
  list(model = svm_fit, prob_valid = p_valid)
}

# --- (C) Random Forest (probability output) ---
fit_predict_rf <- function(train_df, valid_df, predictor_names, class_weight = 1,
                           ntree = 500, mtry = NULL) {
  
  x_train <- train_df %>% dplyr::select(all_of(predictor_names))
  y_train <- factor(train_df$lca, levels = c(0, 1))
  
  x_valid <- valid_df %>% dplyr::select(all_of(predictor_names))
  
  if (is.null(mtry)) mtry <- max(1, floor(sqrt(length(predictor_names))))
  
  mod <- randomForest::randomForest(
    x = x_train,
    y = y_train,
    ntree = ntree,
    mtry = mtry,
    importance = FALSE,
    classwt = c("0" = 1, "1" = class_weight)
  )
  
  prob_mat <- predict(mod, x_valid, type = "prob")
  p_valid <- as.numeric(prob_mat[, "1"])
  list(model = mod, prob_valid = p_valid)
}

# --- (D) XGBoost (probability output) ---
fit_predict_xgb <- function(train_df, valid_df, predictor_names,
                            class_weight = 1,
                            nrounds = 300,
                            max_depth = 3,
                            eta = 0.05,
                            subsample = 0.9,
                            colsample_bytree = 0.9,
                            seed = 0) {
  
  set.seed(seed)
  
  x_train <- train_df %>% dplyr::select(all_of(predictor_names)) %>% as.matrix()
  y_train <- as.numeric(train_df$lca)
  
  x_valid <- valid_df %>% dplyr::select(all_of(predictor_names)) %>% as.matrix()
  
  w_train <- ifelse(train_df$lca == 1, class_weight, 1)
  
  dtrain <- xgboost::xgb.DMatrix(data = x_train, label = y_train, weight = w_train)
  
  params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "logloss",
    max_depth = max_depth,
    eta = eta,
    subsample = subsample,
    colsample_bytree = colsample_bytree
  )
  
  mod <- xgboost::xgb.train(
    params = params,
    data = dtrain,
    nrounds = nrounds,
    verbose = 0
  )
  
  p_valid <- as.numeric(predict(mod, newdata = x_valid))
  list(model = mod, prob_valid = p_valid)
}

############################################################
# 2) Shared evaluator: overall + per cohort
############################################################

evaluate_probs <- function(eval_df, p_eval,
                           threshold = 0.7,
                           n_boot = 1000,
                           seed = 0,
                           cohort_col = "study") {
  
  if (length(p_eval) != nrow(eval_df)) {
    stop(sprintf("Length mismatch: length(p_eval)=%d but nrow(eval_df)=%d",
                 length(p_eval), nrow(eval_df)))
  }
  
  y_eval <- eval_df$lca
  
  auroc <- compute_auroc(y_eval, p_eval)
  auprc <- compute_auprc(y_eval, p_eval)
  
  auroc_ci <- bootstrap_metric_ci(y_eval, p_eval, compute_auroc, n_boot = n_boot, seed = seed)
  auprc_ci <- bootstrap_metric_ci(y_eval, p_eval, compute_auprc, n_boot = n_boot, seed = seed + 1)
  
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
    threshold = threshold,
    sensitivity = thr$sensitivity,
    specificity = thr$specificity,
    ppv = thr$ppv,
    npv = thr$npv
  )
  
  per_cohort <- eval_df %>%
    dplyr::mutate(prob = p_eval) %>%
    dplyr::group_by(.data[[cohort_col]]) %>%
    dplyr::group_modify(function(dat, key) {
      
      y <- dat$lca
      p <- dat$prob
      cohort_name <- as.character(key[[1]])
      
      if (length(unique(y)) < 2) {
        return(data.frame(
          cohort = cohort_name,
          n = nrow(dat),
          auroc = NA, auroc_ci_low = NA, auroc_ci_high = NA,
          auprc = NA, auprc_ci_low = NA, auprc_ci_high = NA,
          threshold = threshold,
          sensitivity = NA, specificity = NA, ppv = NA, npv = NA
        ))
      }
      
      seed_offset <- as.integer(abs(stats::runif(1) * 1e6))
      a <- compute_auroc(y, p)
      pr <- compute_auprc(y, p)
      
      a_ci <- bootstrap_metric_ci(y, p, compute_auroc, n_boot = n_boot, seed = seed + seed_offset)
      pr_ci <- bootstrap_metric_ci(y, p, compute_auprc, n_boot = n_boot, seed = seed + seed_offset + 1)
      
      t <- compute_threshold_metrics(y, p, threshold = threshold)
      
      data.frame(
        cohort = cohort_name,
        n = nrow(dat),
        auroc = a,
        auroc_ci_low = a_ci$ci_low,
        auroc_ci_high = a_ci$ci_high,
        auprc = pr,
        auprc_ci_low = pr_ci$ci_low,
        auprc_ci_high = pr_ci$ci_high,
        threshold = threshold,
        sensitivity = t$sensitivity,
        specificity = t$specificity,
        ppv = t$ppv,
        npv = t$npv
      )
    }) %>%
    dplyr::ungroup()
  
  dplyr::bind_rows(overall_row, per_cohort)
}

############################################################
# 3) One-call runner for one CSV across comparator models
############################################################

run_comparator_models_one_csv <- function(
    csv_path,
    derivation_studies = c("cafpint", "pali", "redvent"),
    validation_studies = c("cpccrn", "chop"),
    threshold = 0.7,
    class_weight = 5.5,
    n_boot = 1000,
    seed = 0
) {
  df <- load_dataset_csv(csv_path)
  
  if (is.factor(df$lca)) df$lca <- as.character(df$lca)
  df$lca <- as.integer(df$lca)
  
  if (!all(df$lca %in% c(0, 1))) stop("lca must be coded 0/1.")
  
  candidate_predictors <- get_predictor_names(df)
  df_imp <- median_impute_all(df, candidate_predictors)
  
  stepwise_res <- select_predictors_stepwise(
    df_imputed = df_imp,
    candidate_predictors = candidate_predictors
  )
  
  predictor_names <- stepwise_res$selected_predictors
  
  message("Comparator models using stepwise-selected predictors (", length(predictor_names), "):")
  message(paste(predictor_names, collapse = ", "))
  
  train_df <- df_imp %>% dplyr::filter(study %in% derivation_studies)
  valid_df <- df_imp %>% dplyr::filter(study %in% validation_studies)
  
  if (nrow(train_df) == 0) stop("No derivation rows found.")
  if (nrow(valid_df) == 0) stop("No validation rows found.")
  
  all_metrics <- list()
  all_preds <- list()
  
  # GLM
  glm_res <- fit_predict_glm(train_df, valid_df, predictor_names, class_weight = class_weight)
  glm_metrics <- evaluate_probs(valid_df, glm_res$prob_valid, threshold, n_boot, seed)
  glm_metrics$model <- "GLM"
  glm_metrics$dataset <- basename(csv_path)
  all_metrics[["GLM"]] <- glm_metrics
  all_preds[["GLM"]] <- data.frame(
    study = valid_df$study, y_true = valid_df$lca, prob = glm_res$prob_valid, model = "GLM"
  )
  
  # SVM
  svm_res <- fit_predict_svm(train_df, valid_df, predictor_names, class_weight = class_weight, seed = seed)
  svm_metrics <- evaluate_probs(valid_df, svm_res$prob_valid, threshold, n_boot, seed)
  svm_metrics$model <- "SVM"
  svm_metrics$dataset <- basename(csv_path)
  all_metrics[["SVM"]] <- svm_metrics
  all_preds[["SVM"]] <- data.frame(
    study = valid_df$study, y_true = valid_df$lca, prob = svm_res$prob_valid, model = "SVM"
  )
  
  # Random Forest
  rf_res <- fit_predict_rf(train_df, valid_df, predictor_names, class_weight = class_weight)
  rf_metrics <- evaluate_probs(valid_df, rf_res$prob_valid, threshold, n_boot, seed)
  rf_metrics$model <- "RandomForest"
  rf_metrics$dataset <- basename(csv_path)
  all_metrics[["RandomForest"]] <- rf_metrics
  all_preds[["RandomForest"]] <- data.frame(
    study = valid_df$study, y_true = valid_df$lca, prob = rf_res$prob_valid, model = "RandomForest"
  )
  
  # XGBoost
  xgb_res <- fit_predict_xgb(train_df, valid_df, predictor_names, class_weight = class_weight, seed = seed)
  xgb_metrics <- evaluate_probs(valid_df, xgb_res$prob_valid, threshold, n_boot, seed)
  xgb_metrics$model <- "XGBoost"
  xgb_metrics$dataset <- basename(csv_path)
  all_metrics[["XGBoost"]] <- xgb_metrics
  all_preds[["XGBoost"]] <- data.frame(
    study = valid_df$study, y_true = valid_df$lca, prob = xgb_res$prob_valid, model = "XGBoost"
  )
  
  metrics_df <- dplyr::bind_rows(all_metrics)
  preds_df <- dplyr::bind_rows(all_preds)
  
  list(
    metrics = metrics_df,
    preds = preds_df,
    predictors = predictor_names
  )
}
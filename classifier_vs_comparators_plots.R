############################################################
# classifier_vs_comparators_plots.R
#
# Purpose:
#   Plot ROC + PR curves comparing:
#     - the main classifier model (ALL_VALIDATION only)
#     - comparator models from comparator_models.R
#
# Inputs:
#   - csv_path: dataset CSV path for classifier predictions
#   - classifier_model: fitted cv.glmnet object from classifier_core.R
#   - classifier_predictors: character vector of predictors actually used
#       by the trained classifier (e.g. res$predictors)
#   - comparator_preds_df: dataframe from comparator_models.R
#       required columns: study, y_true, prob, model
#
# Output:
#   - Combined ROC and PR plots
#
############################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(pROC)
  library(PRROC)
  library(scales)
})

source("classifier_core.R")

############################################################
# 1) Build classifier ALL_VALIDATION predictions
############################################################

prepare_classifier_validation_predictions <- function(
    csv_path,
    classifier_model,
    classifier_predictors,
    validation_studies = c("cpccrn", "chop"),
    classifier_label = "Classifier"
) {
  df <- load_dataset_csv(csv_path)
  
  # Robust lca conversion
  if (is.factor(df$lca)) df$lca <- as.character(df$lca)
  df$lca <- as.integer(df$lca)
  if (!all(df$lca %in% c(0, 1))) stop("lca must be coded 0/1 in the CSV.")
  
  if (missing(classifier_predictors) || length(classifier_predictors) == 0) {
    stop("classifier_predictors must be supplied and non-empty.")
  }
  
  missing_preds <- setdiff(classifier_predictors, colnames(df))
  if (length(missing_preds) > 0) {
    stop(sprintf(
      "These classifier_predictors are missing from the CSV: %s",
      paste(missing_preds, collapse = ", ")
    ))
  }
  
  # Impute all candidate predictors in the CSV so selected predictors are available
  candidate_predictors <- get_predictor_names(df)
  df_imp <- median_impute_all(df, candidate_predictors)
  
  eval_df <- df_imp %>% filter(study %in% validation_studies)
  if (nrow(eval_df) == 0) stop("No rows found for validation_studies in this CSV.")
  
  if (is.factor(eval_df$lca)) eval_df$lca <- as.character(eval_df$lca)
  eval_df$lca <- as.integer(eval_df$lca)
  if (!all(eval_df$lca %in% c(0, 1))) stop("After preprocessing, lca is not 0/1.")
  
  # IMPORTANT: use the actual classifier predictors, not all predictors
  x_eval <- eval_df %>%
    dplyr::select(all_of(classifier_predictors)) %>%
    as.matrix()
  
  p_eval <- as.numeric(
    predict(classifier_model, newx = x_eval, s = "lambda.min", type = "response")
  )
  
  if (length(p_eval) != nrow(eval_df)) {
    stop(sprintf(
      "Prediction length mismatch: got %d preds for %d rows.",
      length(p_eval), nrow(eval_df)
    ))
  }
  
  out <- eval_df %>%
    transmute(
      study = study,
      y_true = lca,
      prob = p_eval,
      model = classifier_label
    )
  
  out
}

############################################################
# 2) Combine classifier + comparator predictions
############################################################

prepare_combined_predictions <- function(
    csv_path,
    classifier_model,
    classifier_predictors,
    comparator_preds_df,
    validation_studies = c("cpccrn", "chop"),
    classifier_label = "Classifier"
) {
  required_cols <- c("study", "y_true", "prob", "model")
  if (!all(required_cols %in% colnames(comparator_preds_df))) {
    stop("comparator_preds_df must contain columns: study, y_true, prob, model")
  }
  
  classifier_df <- prepare_classifier_validation_predictions(
    csv_path = csv_path,
    classifier_model = classifier_model,
    classifier_predictors = classifier_predictors,
    validation_studies = validation_studies,
    classifier_label = classifier_label
  )
  
  comparator_df <- comparator_preds_df %>%
    mutate(
      study = as.character(study),
      y_true = as.integer(y_true),
      prob = as.numeric(prob),
      model = as.character(model)
    )
  
  combined_df <- bind_rows(classifier_df, comparator_df)
  
  model_order <- c(classifier_label, "GLM", "RandomForest", "SVM", "XGBoost")
  present_order <- model_order[model_order %in% unique(combined_df$model)]
  remaining <- setdiff(unique(combined_df$model), present_order)
  combined_df$model <- factor(combined_df$model, levels = c(present_order, remaining))
  
  combined_df
}

############################################################
# 3) ROC helpers
############################################################

make_combined_roc_df <- function(preds_df) {
  preds_df %>%
    group_by(model) %>%
    group_map(function(dat, key) {
      y <- as.integer(dat$y_true)
      p <- as.numeric(dat$prob)
      
      if (length(unique(y)) < 2) {
        return(data.frame())
      }
      
      roc_obj <- pROC::roc(
        response = y,
        predictor = p,
        levels = c(0, 1),
        direction = "<",
        quiet = TRUE
      )
      
      auc_val <- as.numeric(pROC::auc(roc_obj))
      
      data.frame(
        fpr = 1 - roc_obj$specificities,
        sensitivity = roc_obj$sensitivities,
        model = as.character(key$model),
        model_label = paste0(as.character(key$model), " (AUC = ", sprintf("%.3f", auc_val), ")")
      )
    }) %>%
    bind_rows() %>%
    ungroup()
}

plot_classifier_vs_comparators_roc <- function(
    preds_df,
    title = "ROC: Classifier vs comparator models (validation)"
) {
  roc_df <- make_combined_roc_df(preds_df)
  
  if (nrow(roc_df) == 0) {
    stop("No ROC data to plot.")
  }
  
  ggplot(roc_df, aes(x = fpr, y = sensitivity, color = model_label)) +
    geom_path(linewidth = 1.15) +
    geom_abline(
      slope = 1, intercept = 0,
      linetype = "dashed",
      color = "grey55",
      linewidth = 0.8
    ) +
    scale_x_continuous(
      limits = c(0, 1),
      expand = c(0, 0),
      labels = number_format(accuracy = 0.1)
    ) +
    scale_y_continuous(
      limits = c(0, 1),
      expand = c(0, 0),
      labels = number_format(accuracy = 0.1)
    ) +
    coord_equal() +
    theme_bw(base_size = 13) +
    labs(
      title = title,
      x = "False positive rate (1 - specificity)",
      y = "True positive rate (sensitivity)",
      color = "Model"
    ) +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      axis.title = element_text(face = "bold"),
      axis.text = element_text(size = 11),
      legend.position = "right",
      legend.title = element_text(face = "bold"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "grey92", linewidth = 0.3)
    )
}

############################################################
# 4) PR helpers
############################################################

make_combined_pr_df <- function(preds_df) {
  preds_df %>%
    group_by(model) %>%
    group_map(function(dat, key) {
      y <- as.integer(dat$y_true)
      p <- as.numeric(dat$prob)
      
      if (length(unique(y)) < 2) {
        return(data.frame())
      }
      
      pr_obj <- PRROC::pr.curve(
        scores.class0 = p[y == 1],
        scores.class1 = p[y == 0],
        curve = TRUE
      )
      
      auprc_val <- as.numeric(pr_obj$auc.integral)
      pr_curve <- as.data.frame(pr_obj$curve)
      
      if (ncol(pr_curve) < 2) {
        return(data.frame())
      }
      
      colnames(pr_curve)[1:2] <- c("recall", "precision")
      if (ncol(pr_curve) >= 3) colnames(pr_curve)[3] <- "threshold"
      
      pr_curve %>%
        mutate(
          model = as.character(key$model),
          model_label = paste0(as.character(key$model), " (AUPRC = ", sprintf("%.3f", auprc_val), ")")
        ) %>%
        filter(
          !is.na(recall), !is.na(precision),
          recall >= 0, recall <= 1,
          precision >= 0, precision <= 1
        )
    }) %>%
    bind_rows() %>%
    ungroup()
}

plot_classifier_vs_comparators_pr <- function(
    preds_df,
    title = "PR: Classifier vs comparator models (validation)"
) {
  pr_df <- make_combined_pr_df(preds_df)
  
  if (nrow(pr_df) == 0) {
    stop("No PR data to plot.")
  }
  
  prevalence <- mean(as.integer(preds_df$y_true) == 1, na.rm = TRUE)
  
  ggplot(pr_df, aes(x = recall, y = precision, color = model_label)) +
    geom_path(linewidth = 1.15) +
    geom_hline(
      yintercept = prevalence,
      linetype = "dashed",
      color = "grey55",
      linewidth = 0.8
    ) +
    scale_x_continuous(
      limits = c(0, 1),
      expand = c(0, 0),
      labels = number_format(accuracy = 0.1)
    ) +
    scale_y_continuous(
      limits = c(0, 1),
      expand = c(0, 0),
      labels = number_format(accuracy = 0.1)
    ) +
    coord_equal() +
    theme_bw(base_size = 13) +
    labs(
      title = title,
      subtitle = paste0(
        "Dashed horizontal line = positive prevalence baseline (",
        percent(prevalence, accuracy = 0.1), ")"
      ),
      x = "Recall (sensitivity)",
      y = "Precision (PPV)",
      color = "Model"
    ) +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      plot.subtitle = element_text(size = 10),
      axis.title = element_text(face = "bold"),
      axis.text = element_text(size = 11),
      legend.position = "right",
      legend.title = element_text(face = "bold"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "grey92", linewidth = 0.3)
    )
}

############################################################
# 5) Save helper
############################################################

save_classifier_vs_comparators_plots <- function(
    roc_plot,
    pr_plot,
    out_dir = "results/plots",
    prefix = "comparisons",
    width = 7,
    height = 6,
    dpi = 300
) {
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  
  ggplot2::ggsave(
    filename = file.path(out_dir, paste0("classifier_vs_comparators_ROC_", prefix, ".png")),
    plot = roc_plot,
    width = width,
    height = height,
    dpi = dpi
  )
  
  ggplot2::ggsave(
    filename = file.path(out_dir, paste0("classifier_vs_comparators_PR_", prefix, ".png")),
    plot = pr_plot,
    width = width,
    height = height,
    dpi = dpi
  )
}

############################################################
# 6) One-call wrapper
############################################################

plot_classifier_vs_comparators <- function(
    csv_path,
    classifier_model,
    classifier_predictors,
    comparator_preds_df,
    validation_studies = c("cpccrn", "chop"),
    classifier_label = "Classifier",
    out_dir = "results/plots",
    prefix = "comparisons",
    save_plots = TRUE
) {
  combined_df <- prepare_combined_predictions(
    csv_path = csv_path,
    classifier_model = classifier_model,
    classifier_predictors = classifier_predictors,
    comparator_preds_df = comparator_preds_df,
    validation_studies = validation_studies,
    classifier_label = classifier_label
  )
  
  roc_plot <- plot_classifier_vs_comparators_roc(combined_df)
  pr_plot <- plot_classifier_vs_comparators_pr(combined_df)
  
  if (isTRUE(save_plots)) {
    save_classifier_vs_comparators_plots(
      roc_plot = roc_plot,
      pr_plot = pr_plot,
      out_dir = out_dir,
      prefix = prefix
    )
  }
  
  list(
    combined_df = combined_df,
    roc_plot = roc_plot,
    pr_plot = pr_plot
  )
}
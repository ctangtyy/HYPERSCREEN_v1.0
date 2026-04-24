############################################################
# classifier_plots.R
#
# Purpose:
#   Plot ROC + PR curves (overall + per cohort) WITHOUT retraining.
#
# Inputs:
#   - csv_path: path to the dataset CSV
#   - model: a fitted cv.glmnet object (from run_one_csv()$model)
#
# Uses the same preprocessing as classifier_core.R:
#   - identify predictors
#   - median-impute predictors across entire CSV
#
############################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(pROC)
  library(PRROC)
  library(scales)
})

############################################################
# 1) Build an evaluation dataframe with predicted probabilities
############################################################

#' Prepare evaluation predictions for plotting.
#'
#' Statistically:
#'   - We take the already-fit model coefficients (frozen)
#'   - Apply the model to held-out validation cohorts
#'   - Get predicted probability P(lca == 1) for each patient
#'
#' @return a dataframe with: study, lca, prob
prepare_eval_predictions <- function(
    csv_path,
    model,
    classifier_predictors,
    validation_studies = c("cpccrn", "chop")
) {
  df <- load_dataset_csv(csv_path)
  
  # Robust lca conversion (avoid factor -> 1/2)
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
  
  # Impute all candidate predictors so selected predictors are available
  candidate_predictors <- get_predictor_names(df)
  df_imp <- median_impute_all(df, candidate_predictors)
  
  # Keep only validation cohorts for plotting
  eval_df <- df_imp %>% filter(study %in% validation_studies)
  if (nrow(eval_df) == 0) stop("No rows found for validation_studies in this CSV.")
  
  # Re-assert lca after any transformations
  if (is.factor(eval_df$lca)) eval_df$lca <- as.character(eval_df$lca)
  eval_df$lca <- as.integer(eval_df$lca)
  if (!all(eval_df$lca %in% c(0, 1))) stop("After preprocessing, lca is not 0/1.")
  
  # Build X matrix using the actual classifier predictors
  x_eval <- eval_df %>%
    dplyr::select(all_of(classifier_predictors)) %>%
    as.matrix()
  
  # Predict P(class=1)
  p_eval <- as.numeric(predict(model, newx = x_eval, s = "lambda.min", type = "response"))
  
  if (length(p_eval) != nrow(eval_df)) {
    stop(sprintf("Prediction length mismatch: got %d preds for %d rows.", length(p_eval), nrow(eval_df)))
  }
  
  out <- eval_df %>%
    dplyr::select(study, lca) %>%
    dplyr::mutate(prob = p_eval)
  
  return(out)
}

############################################################
# 2) ROC plot (overall + per cohort)
############################################################

# Convert ROC object to standard plotting dataframe:
# x = false positive rate, y = true positive rate
roc_to_df <- function(roc_obj, set_name) {
  data.frame(
    fpr = 1 - roc_obj$specificities,
    sensitivity = roc_obj$sensitivities,
    set = set_name
  )
}

#' Create publication-quality ROC curve plot.
make_roc_plot <- function(pred_df) {
  # Overall ROC
  if (length(unique(pred_df$lca)) < 2) {
    stop("Overall validation set has only one class; ROC is undefined.")
  }
  
  roc_all <- pROC::roc(
    response = pred_df$lca,
    predictor = pred_df$prob,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )
  auc_all <- as.numeric(pROC::auc(roc_all))
  roc_all_df <- roc_to_df(
    roc_all,
    paste0("ALL_VALIDATION (AUC = ", sprintf("%.3f", auc_all), ")")
  )
  
  # Per-cohort ROC curves
  cohort_dfs <- pred_df %>%
    dplyr::group_by(study) %>%
    dplyr::group_map(function(dat, key) {
      if (length(unique(dat$lca)) < 2) {
        return(data.frame())
      }
      
      r <- pROC::roc(
        response = dat$lca,
        predictor = dat$prob,
        levels = c(0, 1),
        direction = "<",
        quiet = TRUE
      )
      auc_val <- as.numeric(pROC::auc(r))
      
      roc_to_df(
        r,
        paste0(as.character(key$study), " (AUC = ", sprintf("%.3f", auc_val), ")")
      )
    }) %>%
    dplyr::bind_rows()
  
  plot_df <- dplyr::bind_rows(roc_all_df, cohort_dfs)
  
  if (nrow(plot_df) == 0) stop("No ROC data to plot (all cohorts single-class?).")
  
  ggplot(plot_df, aes(x = fpr, y = sensitivity, color = set)) +
    geom_path(linewidth = 1.15) +
    geom_abline(
      slope = 1, intercept = 0,
      linetype = "dashed", color = "grey55", linewidth = 0.8
    ) +
    scale_x_continuous(
      limits = c(0, 1),
      expand = c(0, 0),
      labels = scales::number_format(accuracy = 0.1)
    ) +
    scale_y_continuous(
      limits = c(0, 1),
      expand = c(0, 0),
      labels = scales::number_format(accuracy = 0.1)
    ) +
    coord_equal() +
    theme_bw(base_size = 13) +
    labs(
      title = "ROC curves (overall + per cohort)",
      x = "False positive rate (1 - specificity)",
      y = "True positive rate (sensitivity)",
      color = "Curve"
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
# 3) PR curve plot (overall + per cohort)
############################################################

#' Convert PRROC curve output into a ggplot-friendly dataframe.
prroc_to_df <- function(pr, set_name) {
  m <- as.data.frame(pr$curve)
  
  # PRROC typically returns columns: recall, precision, threshold
  # but names vary; enforce at least recall/precision
  if (ncol(m) < 2) stop("PRROC curve output has <2 columns; cannot parse.")
  colnames(m)[1:2] <- c("recall", "precision")
  if (ncol(m) >= 3) colnames(m)[3] <- "threshold"
  
  m$set <- set_name
  
  m %>%
    dplyr::filter(!is.na(recall), !is.na(precision)) %>%
    dplyr::filter(recall >= 0, recall <= 1, precision >= 0, precision <= 1)
}

#' Create publication-quality PR curve plot.
make_pr_plot <- function(pred_df) {
  # Overall PR curve requires at least one positive and one negative
  if (length(unique(pred_df$lca)) < 2) {
    stop("Overall validation set has only one class; PR is undefined.")
  }
  
  # Positive prevalence baseline
  prevalence <- mean(pred_df$lca == 1, na.rm = TRUE)
  
  pr_all <- PRROC::pr.curve(
    scores.class0 = pred_df$prob[pred_df$lca == 1],  # positives
    scores.class1 = pred_df$prob[pred_df$lca == 0],  # negatives
    curve = TRUE
  )
  auprc_all <- as.numeric(pr_all$auc.integral)
  pr_all_df <- prroc_to_df(
    pr_all,
    paste0("ALL_VALIDATION (AUPRC = ", sprintf("%.3f", auprc_all), ")")
  )
  
  cohort_pr_dfs <- pred_df %>%
    dplyr::group_by(study) %>%
    dplyr::group_map(function(dat, key) {
      if (length(unique(dat$lca)) < 2) {
        return(data.frame())
      }
      
      pr <- PRROC::pr.curve(
        scores.class0 = dat$prob[dat$lca == 1],
        scores.class1 = dat$prob[dat$lca == 0],
        curve = TRUE
      )
      auprc_val <- as.numeric(pr$auc.integral)
      
      prroc_to_df(
        pr,
        paste0(as.character(key$study), " (AUPRC = ", sprintf("%.3f", auprc_val), ")")
      )
    }) %>%
    dplyr::bind_rows()
  
  plot_df <- dplyr::bind_rows(pr_all_df, cohort_pr_dfs)
  
  if (nrow(plot_df) == 0) stop("No PR data to plot (all cohorts single-class?).")
  
  ggplot(plot_df, aes(x = recall, y = precision, color = set)) +
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
      labels = scales::number_format(accuracy = 0.1)
    ) +
    scale_y_continuous(
      limits = c(0, 1),
      expand = c(0, 0),
      labels = scales::number_format(accuracy = 0.1)
    ) +
    coord_equal() +
    theme_bw(base_size = 13) +
    labs(
      title = "Precision–Recall curves (overall + per cohort)",
      subtitle = paste0(
        "Dashed horizontal line = positive prevalence baseline (",
        scales::percent(prevalence, accuracy = 0.1), ")"
      ),
      x = "Recall (sensitivity)",
      y = "Precision (PPV)",
      color = "Curve"
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
# 4) Save plots helper
############################################################

save_classifier_plots <- function(roc_plot, pr_plot, out_dir = ".", prefix = "classifier") {
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  
  ggplot2::ggsave(
    filename = file.path(out_dir, paste0("classifier_ROC_plot_", prefix, ".png")),
    plot = roc_plot,
    width = 7,
    height = 5,
    dpi = 300
  )
  
  ggplot2::ggsave(
    filename = file.path(out_dir, paste0("classifier_PR_plot_", prefix, ".png")),
    plot = pr_plot,
    width = 7,
    height = 5,
    dpi = 300
  )
}

############################################################
# 4b) Publication-quality confusion matrices
############################################################

compute_confusion_counts <- function(pred_df, threshold = 0.7) {
  stopifnot(all(c("lca", "prob") %in% colnames(pred_df)))
  
  y_true <- as.integer(pred_df$lca)
  y_pred <- as.integer(pred_df$prob >= threshold)
  
  list(
    tn = sum(y_true == 0 & y_pred == 0, na.rm = TRUE),
    fp = sum(y_true == 0 & y_pred == 1, na.rm = TRUE),
    fn = sum(y_true == 1 & y_pred == 0, na.rm = TRUE),
    tp = sum(y_true == 1 & y_pred == 1, na.rm = TRUE)
  )
}

plot_confusion_matrix_classifier <- function(tn, fp, fn, tp, title, subtitle) {
  cm <- data.frame(
    True = c(0, 0, 1, 1),
    Pred = c(0, 1, 0, 1),
    count = c(tn, fp, fn, tp)
  )
  
  cm <- cm %>%
    dplyr::group_by(True) %>%
    dplyr::mutate(
      row_total = sum(count),
      row_percent = ifelse(row_total == 0, NA_real_, count / row_total)
    ) %>%
    dplyr::ungroup()
  
  cm$True <- factor(cm$True, levels = c(1, 0))
  cm$Pred <- factor(cm$Pred, levels = c(0, 1))
  
  cm <- cm %>%
    dplyr::mutate(
      label = paste0(
        count,
        "\n(",
        scales::percent(row_percent, accuracy = 0.1),
        ")"
      )
    )
  
  ggplot(cm, aes(x = Pred, y = True, fill = row_percent)) +
    geom_tile(color = "white", linewidth = 0.8) +
    geom_text(aes(label = label), size = 5) +
    scale_fill_gradient(
      low = "#deebf7",
      high = "#08519c",
      limits = c(0, 1),
      na.value = "grey90"
    ) +
    labs(
      title = title,
      subtitle = subtitle,
      x = "Predicted class",
      y = "True class",
      fill = "Row %"
    ) +
    coord_fixed() +
    theme_bw(base_size = 13) +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      plot.subtitle = element_text(size = 10),
      axis.title = element_text(face = "bold"),
      axis.text = element_text(size = 12),
      panel.grid = element_blank(),
      legend.position = "right",
      legend.title = element_text(size = 10),
      legend.text = element_text(size = 9)
    )
}

save_classifier_confusion_matrices_png <- function(pred_df,
                                                   threshold = 0.7,
                                                   out_dir = "results/confusion_matrices",
                                                   prefix = "classifier",
                                                   width = 8,
                                                   height = 5.5,
                                                   dpi = 300) {
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  
  # ALL_VALIDATION
  cc_all <- compute_confusion_counts(pred_df, threshold = threshold)
  p_all <- plot_confusion_matrix_classifier(
    tn = cc_all$tn, fp = cc_all$fp, fn = cc_all$fn, tp = cc_all$tp,
    title = paste0("Confusion Matrix — ", prefix, " (ALL_VALIDATION)"),
    subtitle = paste0("Threshold = ", threshold)
  )
  
  ggplot2::ggsave(
    filename = file.path(out_dir, paste0("confmat_classifier_ALL_VALIDATION_", prefix, ".png")),
    plot = p_all,
    width = width,
    height = height,
    dpi = dpi
  )
  
  # Per cohort
  cohorts <- sort(unique(pred_df$study))
  plots <- list(ALL_VALIDATION = p_all)
  
  for (coh in cohorts) {
    df_c <- pred_df %>% dplyr::filter(study == coh)
    cc <- compute_confusion_counts(df_c, threshold = threshold)
    
    p <- plot_confusion_matrix_classifier(
      tn = cc$tn, fp = cc$fp, fn = cc$fn, tp = cc$tp,
      title = paste0("Confusion Matrix — ", prefix, " (", coh, ")"),
      subtitle = paste0("Threshold = ", threshold)
    )
    
    ggplot2::ggsave(
      filename = file.path(out_dir, paste0("confmat_classifier_", toupper(coh), "_", prefix, ".png")),
      plot = p,
      width = width,
      height = height,
      dpi = dpi
    )
    
    plots[[coh]] <- p
  }
  
  invisible(plots)
}

############################################################
# 5) One-call plotting wrapper (no retraining)
############################################################

plot_from_trained_model <- function(
    csv_path,
    model,
    classifier_predictors,
    validation_studies = c("cpccrn", "chop"),
    out_dir = "results/plots",
    prefix = NULL,
    threshold = 0.7,
    save_confmats = FALSE,
    confmat_dir = "results/confusion_matrices"
) {
  if (is.null(prefix)) prefix <- tools::file_path_sans_ext(basename(csv_path))
  if (is.null(confmat_dir)) confmat_dir <- file.path(out_dir, "confusion_matrices")
  
  pred_df <- prepare_eval_predictions(
    csv_path = csv_path,
    model = model,
    classifier_predictors = classifier_predictors,
    validation_studies = validation_studies
  )
  
  roc_plot <- make_roc_plot(pred_df)
  pr_plot <- make_pr_plot(pred_df)
  
  save_classifier_plots(roc_plot, pr_plot, out_dir = out_dir, prefix = prefix)
  
  if (isTRUE(save_confmats)) {
    save_classifier_confusion_matrices_png(
      pred_df = pred_df,
      threshold = threshold,
      out_dir = confmat_dir,
      prefix = prefix
    )
  }
  
  return(list(pred_df = pred_df, roc_plot = roc_plot, pr_plot = pr_plot))
}  
  
############################################################
# 6) Calibration plots (Step A)
############################################################

plot_stepA_calibration_set <- function(pred_df,
                                       study_value = NULL,
                                       n_bins = 10) {
  if (!all(c("study", "lca", "prob") %in% colnames(pred_df))) {
    stop("pred_df must contain columns: study, lca, prob")
  }

  if (is.null(study_value)) {
    df_use <- pred_df
    title <- "Calibration Curve — ALL validation"
  } else {
    df_use <- pred_df %>% dplyr::filter(study == study_value)

    if (nrow(df_use) == 0) {
      stop(sprintf("No rows found for study '%s' in pred_df.", study_value))
    }

    title <- paste0("Calibration Curve — ", toupper(study_value))
  }

  calib_df <- make_calibration_df(
    df = df_use,
    outcome_col = "lca",
    prob_col = "prob",
    n_bins = n_bins
  )

  plot_calibration_curve(
    calib_df = calib_df,
    title = title,
    subtitle = "Observed event rate with 95% binomial confidence band"
  )
}

save_stepA_calibration_plots <- function(pred_df,
                                         out_dir = "results/calibration",
                                         prefix = "classifier",
                                         n_bins_all = 10,
                                         n_bins_per_cohort = 8,
                                         width = 6,
                                         height = 5,
                                         dpi = 300) {
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

  plots <- list()

  # ALL validation
  p_all <- plot_stepA_calibration_set(
    pred_df = pred_df,
    study_value = NULL,
    n_bins = n_bins_all
  )

  save_calibration_plot(
    p_all,
    file.path(out_dir, paste0("calibration_ALL_VALIDATION_", prefix, ".png")),
    width = width,
    height = height,
    dpi = dpi
  )

  plots[["ALL_VALIDATION"]] <- p_all

  # Per cohort: specifically CHOP and CPCCRN if present
  for (coh in c("chop", "cpccrn")) {
    if (coh %in% unique(pred_df$study)) {
      p <- plot_stepA_calibration_set(
        pred_df = pred_df,
        study_value = coh,
        n_bins = n_bins_per_cohort
      )

      save_calibration_plot(
        p,
        file.path(out_dir, paste0("calibration_", toupper(coh), "_", prefix, ".png")),
        width = width,
        height = height,
        dpi = dpi
      )

      plots[[toupper(coh)]] <- p
    }
  }

  invisible(plots)
}
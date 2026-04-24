############################################################
# classifier_stepB_plots.R
#
# Purpose:
#   Plot ROC + PR curves + confusion matrix for Step B
#   using out-of-fold CV predictions from classifier_stepB_core.R
#
# Inputs:
#   - pred_df: dataframe containing at least:
#       lca  = true binary outcome (0/1)
#       prob = predicted probability for class 1
#     optional:
#       study, id, idstudy
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
# 1) Validation checks
############################################################

validate_stepB_pred_df <- function(pred_df) {
  required_cols <- c("lca", "prob")
  missing_cols <- setdiff(required_cols, colnames(pred_df))
  
  if (length(missing_cols) > 0) {
    stop(sprintf(
      "pred_df is missing required columns: %s",
      paste(missing_cols, collapse = ", ")
    ))
  }
  
  pred_df <- pred_df %>%
    dplyr::mutate(
      lca = as.integer(lca),
      prob = as.numeric(prob)
    )
  
  if (!all(pred_df$lca %in% c(0, 1))) {
    stop("pred_df$lca must be coded as 0/1.")
  }
  
  if (any(is.na(pred_df$prob))) {
    stop("pred_df$prob contains NA values.")
  }
  
  pred_df
}

############################################################
# 2) ROC plot
############################################################

make_stepB_roc_df <- function(pred_df) {
  pred_df <- validate_stepB_pred_df(pred_df)
  
  if (length(unique(pred_df$lca)) < 2) {
    stop("ROC is undefined because pred_df$lca has only one class.")
  }
  
  roc_obj <- pROC::roc(
    response = pred_df$lca,
    predictor = pred_df$prob,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )
  
  auc_val <- as.numeric(pROC::auc(roc_obj))
  
  data.frame(
    fpr = 1 - roc_obj$specificities,
    sensitivity = roc_obj$sensitivities,
    curve_label = paste0("ALL_DATA_CV (AUC = ", sprintf("%.3f", auc_val), ")")
  )
}

plot_stepB_roc <- function(pred_df,
                           title = "ROC curve: Step B cross-validated model") {
  roc_df <- make_stepB_roc_df(pred_df)
  
  ggplot(roc_df, aes(x = fpr, y = sensitivity, color = curve_label)) +
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
      title = title,
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
# 3) PR plot
############################################################

make_stepB_pr_df <- function(pred_df) {
  pred_df <- validate_stepB_pred_df(pred_df)
  
  if (length(unique(pred_df$lca)) < 2) {
    stop("PR curve is undefined because pred_df$lca has only one class.")
  }
  
  pr_obj <- PRROC::pr.curve(
    scores.class0 = pred_df$prob[pred_df$lca == 1],
    scores.class1 = pred_df$prob[pred_df$lca == 0],
    curve = TRUE
  )
  
  auprc_val <- as.numeric(pr_obj$auc.integral)
  pr_curve <- as.data.frame(pr_obj$curve)
  
  if (ncol(pr_curve) < 2) {
    stop("PRROC curve output has fewer than 2 columns.")
  }
  
  colnames(pr_curve)[1:2] <- c("recall", "precision")
  if (ncol(pr_curve) >= 3) colnames(pr_curve)[3] <- "threshold"
  
  pr_curve %>%
    dplyr::mutate(
      curve_label = paste0("ALL_DATA_CV (AUPRC = ", sprintf("%.3f", auprc_val), ")")
    ) %>%
    dplyr::filter(
      !is.na(recall), !is.na(precision),
      recall >= 0, recall <= 1,
      precision >= 0, precision <= 1
    )
}

plot_stepB_pr <- function(pred_df,
                          title = "Precision–Recall curve: Step B cross-validated model") {
  pred_df <- validate_stepB_pred_df(pred_df)
  pr_df <- make_stepB_pr_df(pred_df)
  
  prevalence <- mean(pred_df$lca == 1, na.rm = TRUE)
  
  ggplot(pr_df, aes(x = recall, y = precision, color = curve_label)) +
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
      title = title,
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
# 4) Confusion matrix helpers
############################################################

compute_stepB_confusion_counts <- function(pred_df, threshold = 0.7) {
  pred_df <- validate_stepB_pred_df(pred_df)
  
  y_true <- as.integer(pred_df$lca)
  y_pred <- as.integer(pred_df$prob >= threshold)
  
  list(
    tn = sum(y_true == 0 & y_pred == 0, na.rm = TRUE),
    fp = sum(y_true == 0 & y_pred == 1, na.rm = TRUE),
    fn = sum(y_true == 1 & y_pred == 0, na.rm = TRUE),
    tp = sum(y_true == 1 & y_pred == 1, na.rm = TRUE)
  )
}

compute_stepB_confusion_metrics <- function(tn, fp, fn, tp) {
  total <- tn + fp + fn + tp
  
  sensitivity <- ifelse(tp + fn == 0, NA_real_, tp / (tp + fn))
  specificity <- ifelse(tn + fp == 0, NA_real_, tn / (tn + fp))
  ppv <- ifelse(tp + fp == 0, NA_real_, tp / (tp + fp))
  npv <- ifelse(tn + fn == 0, NA_real_, tn / (tn + fn))
  accuracy <- ifelse(total == 0, NA_real_, (tp + tn) / total)
  
  data.frame(
    sensitivity = sensitivity,
    specificity = specificity,
    ppv = ppv,
    npv = npv,
    accuracy = accuracy
  )
}

plot_stepB_confusion_matrix <- function(pred_df,
                                        threshold = 0.7,
                                        title = "Confusion Matrix — Step B") {
  cc <- compute_stepB_confusion_counts(pred_df, threshold = threshold)
  met <- compute_stepB_confusion_metrics(cc$tn, cc$fp, cc$fn, cc$tp)
  
  cm <- data.frame(
    True = c(0, 0, 1, 1),
    Pred = c(0, 1, 0, 1),
    count = c(cc$tn, cc$fp, cc$fn, cc$tp)
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
  
  metrics_line <- paste0(
    "Threshold = ", threshold,
    "   |   Accuracy = ", scales::percent(met$accuracy, accuracy = 0.1),
    "   |   Sensitivity = ", scales::percent(met$sensitivity, accuracy = 0.1),
    "   |   Specificity = ", scales::percent(met$specificity, accuracy = 0.1),
    "   |   PPV = ", scales::percent(met$ppv, accuracy = 0.1),
    "   |   NPV = ", scales::percent(met$npv, accuracy = 0.1)
  )
  
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
      subtitle = metrics_line,
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

############################################################
# 5) Save helpers
############################################################

save_stepB_plots <- function(roc_plot,
                             pr_plot,
                             out_dir = "results/stepB/plots",
                             prefix = "stepB",
                             width = 7,
                             height = 5,
                             dpi = 300) {
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  
  ggplot2::ggsave(
    filename = file.path(out_dir, paste0("stepB_ROC_plot_", prefix, ".png")),
    plot = roc_plot,
    width = width,
    height = height,
    dpi = dpi
  )
  
  ggplot2::ggsave(
    filename = file.path(out_dir, paste0("stepB_PR_plot_", prefix, ".png")),
    plot = pr_plot,
    width = width,
    height = height,
    dpi = dpi
  )
}

save_stepB_confusion_matrix_png <- function(pred_df,
                                            threshold = 0.7,
                                            out_dir = "results/stepB/confusion_matrices",
                                            prefix = "stepB",
                                            width = 8,
                                            height = 5.5,
                                            dpi = 300) {
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  
  p <- plot_stepB_confusion_matrix(
    pred_df = pred_df,
    threshold = threshold,
    title = paste0("Confusion Matrix — ", prefix, " (ALL_DATA_CV)")
  )
  
  ggplot2::ggsave(
    filename = file.path(out_dir, paste0("confmat_stepB_ALL_DATA_CV_", prefix, ".png")),
    plot = p,
    width = width,
    height = height,
    dpi = dpi
  )
  
  invisible(p)
}

############################################################
# 6) One-call wrapper
############################################################

plot_stepB_from_predictions <- function(pred_df,
                                        threshold = 0.7,
                                        out_dir = "results/stepB/plots",
                                        confmat_dir = "results/stepB/confusion_matrices",
                                        prefix = "stepB",
                                        save_plots = TRUE,
                                        save_confmat = TRUE) {
  pred_df <- validate_stepB_pred_df(pred_df)
  
  roc_plot <- plot_stepB_roc(pred_df)
  pr_plot <- plot_stepB_pr(pred_df)
  
  if (isTRUE(save_plots)) {
    save_stepB_plots(
      roc_plot = roc_plot,
      pr_plot = pr_plot,
      out_dir = out_dir,
      prefix = prefix
    )
  }
  
  confmat_plot <- NULL
  if (isTRUE(save_confmat)) {
    confmat_plot <- save_stepB_confusion_matrix_png(
      pred_df = pred_df,
      threshold = threshold,
      out_dir = confmat_dir,
      prefix = prefix
    )
  }
  
  list(
    roc_plot = roc_plot,
    pr_plot = pr_plot,
    confmat_plot = confmat_plot
  )
}

############################################################
# 7) Calibration plots (Step B)
############################################################

plot_stepB_calibration_set <- function(pred_df,
                                       study_value = NULL,
                                       n_bins = 10) {
  if (!all(c("study", "lca", "prob") %in% colnames(pred_df))) {
    stop("pred_df must contain columns: study, lca, prob")
  }
  
  if (is.null(study_value)) {
    df_use <- pred_df
    title <- "Calibration Curve — ALL data"
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

save_stepB_calibration_plots <- function(pred_df,
                                         out_dir = "results/stepB/calibration",
                                         prefix = "stepB",
                                         n_bins_all = 10,
                                         n_bins_per_cohort = 8,
                                         width = 6,
                                         height = 5,
                                         dpi = 300) {
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  
  plots <- list()
  
  # ALL data
  p_all <- plot_stepB_calibration_set(
    pred_df = pred_df,
    study_value = NULL,
    n_bins = n_bins_all
  )
  
  save_calibration_plot(
    p_all,
    file.path(out_dir, paste0("calibration_ALL_DATA_", prefix, ".png")),
    width = width,
    height = height,
    dpi = dpi
  )
  
  plots[["ALL_DATA"]] <- p_all
  
  # Per cohort
  cohorts <- sort(unique(pred_df$study))
  
  for (coh in cohorts) {
    p <- plot_stepB_calibration_set(
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
  
  invisible(plots)
}
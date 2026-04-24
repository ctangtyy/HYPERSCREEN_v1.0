############################################################
# comparator_plots.R
#
# Purpose:
#   Plot ROC + PR curves comparing multiple models, using
#   the predictions returned by comparator_models.R
#
# Inputs:
#   preds_df with columns: study, y_true (0/1), prob, model
#
############################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(pROC)
  library(tidyr)
  library(scales)
  library(PRROC)
})

############################################################
# 1) ROC curves
############################################################

# Build ROC curve data for each model
make_roc_df <- function(preds_df) {
  preds_df %>%
    group_by(model) %>%
    group_map(function(dat, key) {
      y <- as.integer(dat$y_true)
      p <- as.numeric(dat$prob)
      
      # Skip if only one class present (ROC undefined)
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
      
      out <- data.frame(
        fpr = 1 - roc_obj$specificities,
        sensitivity = roc_obj$sensitivities
      )
      
      out$model <- as.character(key$model)
      out$model_label <- paste0(as.character(key$model), " (AUC = ", sprintf("%.3f", auc_val), ")")
      out
    }) %>%
    bind_rows() %>%
    ungroup()
}

plot_roc_compare <- function(preds_df,
                             title = "ROC: Comparator models (validation)") {
  roc_df <- make_roc_df(preds_df)
  
  if (nrow(roc_df) == 0) {
    stop("No ROC data to plot (possibly all models had single-class y_true).")
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
# 2) PR curves
############################################################

# Build PR curve data for each model using PRROC
make_pr_df <- function(preds_df) {
  preds_df %>%
    group_by(model) %>%
    group_map(function(dat, key) {
      y <- as.integer(dat$y_true)
      p <- as.numeric(dat$prob)
      
      # PR undefined / not meaningful if only one class present
      if (length(unique(y)) < 2) {
        return(data.frame())
      }
      
      pr_obj <- PRROC::pr.curve(
        scores.class0 = p[y == 1],  # positives
        scores.class1 = p[y == 0],  # negatives
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

plot_pr_compare <- function(preds_df,
                            title = "Precision–Recall: Comparator models (validation)") {
  pr_df <- make_pr_df(preds_df)
  
  if (nrow(pr_df) == 0) {
    stop("No PR data to plot (possibly all models had single-class y_true).")
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
# 3) Save ROC + PR plots
############################################################

save_comparator_plots <- function(roc_plot, pr_plot,
                                  out_dir = ".",
                                  prefix = "comparators",
                                  width = 7,
                                  height = 6,
                                  dpi = 300) {
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  
  ggplot2::ggsave(
    filename = file.path(out_dir, paste0("comparator_ROC_plot_", prefix, ".png")),
    plot = roc_plot,
    width = width,
    height = height,
    dpi = dpi
  )
  
  ggplot2::ggsave(
    filename = file.path(out_dir, paste0("comparator_PR_plot_", prefix, ".png")),
    plot = pr_plot,
    width = width,
    height = height,
    dpi = dpi
  )
}

############################################################
# 4) Confusion matrix helpers
############################################################

# Create confusion matrix counts from y_true and predicted class
confusion_counts <- function(y_true, y_pred) {
  y_true <- as.integer(y_true)
  y_pred <- as.integer(y_pred)
  
  tp <- sum(y_true == 1 & y_pred == 1, na.rm = TRUE)
  tn <- sum(y_true == 0 & y_pred == 0, na.rm = TRUE)
  fp <- sum(y_true == 0 & y_pred == 1, na.rm = TRUE)
  fn <- sum(y_true == 1 & y_pred == 0, na.rm = TRUE)
  
  data.frame(tn = tn, fp = fp, fn = fn, tp = tp)
}

# Compute common derived metrics from confusion counts
confusion_metrics <- function(tn, fp, fn, tp) {
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

# Print confusion matrices for each comparator model at a threshold
print_confusion_matrices <- function(preds_df, threshold = 0.7) {
  stopifnot(all(c("y_true", "prob", "model") %in% colnames(preds_df)))
  
  res <- preds_df %>%
    mutate(
      y_true = as.integer(y_true),
      y_pred = as.integer(prob >= threshold)
    ) %>%
    group_by(model) %>%
    group_map(function(dat, key) {
      cc <- confusion_counts(dat$y_true, dat$y_pred)
      met <- confusion_metrics(cc$tn, cc$fp, cc$fn, cc$tp)
      
      out <- cbind(
        data.frame(
          model = as.character(key$model),
          threshold = threshold,
          n = nrow(dat)
        ),
        cc,
        met
      )
      
      cat("\n==============================\n")
      cat("Model:", out$model, "\n")
      cat("Threshold:", out$threshold, "\n")
      cat("N:", out$n, "\n\n")
      
      cat("Confusion matrix (rows = true, cols = predicted)\n")
      cm <- matrix(
        c(out$tn, out$fp,
          out$fn, out$tp),
        nrow = 2, byrow = TRUE,
        dimnames = list(True = c("0", "1"), Pred = c("0", "1"))
      )
      print(cm)
      
      cat(sprintf("\nSensitivity: %.3f\n", out$sensitivity))
      cat(sprintf("Specificity: %.3f\n", out$specificity))
      cat(sprintf("PPV:         %.3f\n", out$ppv))
      cat(sprintf("NPV:         %.3f\n", out$npv))
      cat(sprintf("Accuracy:    %.3f\n", out$accuracy))
      
      out
    }) %>%
    bind_rows()
  
  invisible(res)
}

# Build confusion matrix dataframe
make_confusion_df <- function(preds_df, threshold = 0.7) {
  preds_df %>%
    mutate(
      y_true = as.integer(y_true),
      y_pred = as.integer(prob >= threshold)
    ) %>%
    group_by(model) %>%
    summarise(
      tn = sum(y_true == 0 & y_pred == 0, na.rm = TRUE),
      fp = sum(y_true == 0 & y_pred == 1, na.rm = TRUE),
      fn = sum(y_true == 1 & y_pred == 0, na.rm = TRUE),
      tp = sum(y_true == 1 & y_pred == 1, na.rm = TRUE),
      .groups = "drop"
    )
}

############################################################
# 5) Publication-quality confusion matrix plotting
############################################################

plot_confusion_matrix <- function(tn, fp, fn, tp, model_name, threshold) {
  cm <- data.frame(
    True = c(0, 0, 1, 1),
    Pred = c(0, 1, 0, 1),
    count = c(tn, fp, fn, tp)
  )
  
  # Row-normalized percentages
  cm <- cm %>%
    group_by(True) %>%
    mutate(
      row_total = sum(count),
      row_percent = ifelse(row_total == 0, NA_real_, count / row_total)
    ) %>%
    ungroup()
  
  # Display order: top row = true 0, bottom row = true 1
  cm$True <- factor(cm$True, levels = c(1, 0))
  cm$Pred <- factor(cm$Pred, levels = c(0, 1))
  
  met <- confusion_metrics(tn, fp, fn, tp)
  
  metrics_line <- paste0(
    "Accuracy = ", percent(met$accuracy, accuracy = 0.1),
    "   |   Sensitivity = ", percent(met$sensitivity, accuracy = 0.1),
    "   |   Specificity = ", percent(met$specificity, accuracy = 0.1),
    "   |   PPV = ", percent(met$ppv, accuracy = 0.1),
    "   |   NPV = ", percent(met$npv, accuracy = 0.1)
  )
  
  cm <- cm %>%
    mutate(
      label = paste0(
        count,
        "\n(",
        percent(row_percent, accuracy = 0.1),
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
      title = paste0("Confusion Matrix — ", model_name),
      subtitle = paste0("Threshold = ", threshold, "\n", metrics_line),
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
# 6) Save confusion matrices as PNGs
############################################################

save_confusion_matrices_png <- function(preds_df,
                                        threshold = 0.7,
                                        out_dir = "results/confusion_matrices",
                                        prefix = "comparators",
                                        width = 8,
                                        height = 5.5,
                                        dpi = 300) {
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  
  cm_df <- make_confusion_df(preds_df, threshold)
  plots <- list()
  
  for (i in seq_len(nrow(cm_df))) {
    row <- cm_df[i, ]
    
    p <- plot_confusion_matrix(
      tn = row$tn,
      fp = row$fp,
      fn = row$fn,
      tp = row$tp,
      model_name = paste0(row$model, " (", prefix, ")"),
      threshold = threshold
    )
    
    file_path <- file.path(
      out_dir,
      paste0("confusion_matrix_", row$model, "_", prefix, ".png")
    )
    
    ggsave(file_path, p, width = width, height = height, dpi = dpi)
    
    plots[[as.character(row$model)]] <- p
  }
  
  invisible(plots)
}
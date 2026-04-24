############################################################
# calibration_helpers.R
#
# Purpose:
#   Shared helpers for calibration curves using:
#     - binned predicted probabilities
#     - observed event rates
#     - exact binomial confidence intervals
#
############################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(scales)
})

############################################################
# 1) Validation
############################################################

validate_calibration_input <- function(df, outcome_col, prob_col) {
  required_cols <- c(outcome_col, prob_col)
  missing_cols <- setdiff(required_cols, colnames(df))
  
  if (length(missing_cols) > 0) {
    stop(sprintf(
      "Calibration input is missing required columns: %s",
      paste(missing_cols, collapse = ", ")
    ))
  }
  
  df <- df %>%
    dplyr::mutate(
      .outcome = as.integer(.data[[outcome_col]]),
      .prob = as.numeric(.data[[prob_col]])
    )
  
  if (!all(df$.outcome %in% c(0, 1))) {
    stop(sprintf("Column '%s' must be coded 0/1.", outcome_col))
  }
  
  if (any(is.na(df$.outcome)) || any(is.na(df$.prob))) {
    stop("Calibration input contains NA values in outcome or probability.")
  }
  
  if (nrow(df) == 0) {
    stop("Calibration input has 0 rows.")
  }
  
  df
}

############################################################
# 2) Bin + summarize one dataset
############################################################

make_calibration_df <- function(df,
                                outcome_col,
                                prob_col = "prob",
                                n_bins = 10,
                                conf_level = 0.95) {
  df <- validate_calibration_input(df, outcome_col = outcome_col, prob_col = prob_col)
  
  probs_seq <- seq(0, 1, length.out = n_bins + 1)
  
  # fixed-width probability bins
  breaks <- seq(0, 1, length.out = n_bins + 1)
  
  df <- df %>%
    mutate(
      .bin = cut(.prob, breaks = breaks, include.lowest = TRUE)
    )
  
  calib <- df %>%
    dplyr::group_by(.bin) %>%
    dplyr::summarise(
      n = dplyr::n(),
      positives = sum(.outcome),
      mean_pred = mean(.prob),
      observed = mean(.outcome),
      .groups = "drop"
    ) %>%
    dplyr::filter(n > 0)
  
  ci_list <- lapply(seq_len(nrow(calib)), function(i) {
    bt <- stats::binom.test(
      x = calib$positives[i],
      n = calib$n[i],
      conf.level = conf_level
    )
    
    data.frame(
      ci_low = bt$conf.int[1],
      ci_high = bt$conf.int[2]
    )
  })
  
  ci_df <- dplyr::bind_rows(ci_list)
  
  dplyr::bind_cols(calib, ci_df)
}

############################################################
# 3) Plot one calibration curve
############################################################

plot_calibration_curve <- function(calib_df,
                                   title = "Calibration Curve",
                                   subtitle = "Observed event rate with 95% binomial confidence band") {
  ggplot(calib_df, aes(x = mean_pred, y = observed)) +
    geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.20) +
    geom_line(linewidth = 1.1) +
    geom_point(size = 2) +
    geom_abline(
      slope = 1, intercept = 0,
      linetype = "dashed",
      color = "grey55",
      linewidth = 0.8
    ) +
    scale_x_continuous(
      limits = c(0, 1),
      labels = scales::number_format(accuracy = 0.1)
    ) +
    scale_y_continuous(
      limits = c(0, 1),
      labels = scales::number_format(accuracy = 0.1)
    ) +
    coord_equal() +
    theme_bw(base_size = 13) +
    labs(
      title = title,
      subtitle = subtitle,
      x = "Predicted Probabilities",
      y = "True Probability"
    ) +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      plot.subtitle = element_text(size = 10),
      axis.title = element_text(face = "bold"),
      axis.text = element_text(size = 11),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "grey92", linewidth = 0.3)
    )
}

############################################################
# 4) Save one calibration plot
############################################################

save_calibration_plot <- function(plot_obj,
                                  file_path,
                                  width = 6,
                                  height = 5,
                                  dpi = 300) {
  ggplot2::ggsave(
    filename = file_path,
    plot = plot_obj,
    width = width,
    height = height,
    dpi = dpi
  )
}
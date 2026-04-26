############################################################
# package_versions.R
#
# Purpose:
#   Save the R version, package versions, and session info used
#   to run the pediatric critical care classifier pipeline.
#
# Usage:
#   source("package_versions.R")
#
# Outputs:
#   - package_versions.csv
#   - sessionInfo.txt
############################################################

required_packages <- c(
  "data.table",
  "dplyr",
  "glmnet",
  "pROC",
  "PRROC",
  "olsrr",
  "ggplot2",
  "scales",
  "e1071",
  "randomForest",
  "xgboost",
  "tidyr"
)

installed <- rownames(installed.packages())
missing_packages <- setdiff(required_packages, installed)

if (length(missing_packages) > 0) {
  message("Installing missing packages: ", paste(missing_packages, collapse = ", "))
  install.packages(missing_packages)
}

package_versions <- data.frame(
  package = required_packages,
  version = vapply(required_packages, function(pkg) {
    as.character(utils::packageVersion(pkg))
  }, character(1)),
  stringsAsFactors = FALSE
)

package_versions <- rbind(
  data.frame(package = "R", version = R.version.string, stringsAsFactors = FALSE),
  package_versions
)

write.csv(package_versions, "package_versions.csv", row.names = FALSE)

sink("sessionInfo.txt")
print(sessionInfo())
sink()

message("Wrote package_versions.csv and sessionInfo.txt")

# Optional renv workflow:
# install.packages("renv")
# renv::init()
# renv::snapshot()

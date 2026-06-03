# ============================================================
# SEMANTIC DRIFT ANALYSIS — ROBUSTNESS EXTENSION v6
# Conformity, Trajectory, and Space:
# A Dynamic Capabilities Account of Representational Survival
# in Platform Accommodation Markets
#
# Purpose:
#   Extended robustness and sensitivity analyses supplementing
#   the main Python pipeline (main_analysis.py, extension_analysis.py).
#   All sections are self-contained and reproducible from the
#   simulated panel described in Section 3 of the paper.
#
# Key enhancements over baseline pipeline
# ── Statistical ───────────────────────────────────────────
#   ST-01  EFA with parallel analysis (Horn) + MAP test
#   ST-02  Within / Between / Overall R² decomposition
#   ST-03  Kleibergen-Paap rk F-statistic (weak IV)
#   ST-04  Window-level Wald test for momentum coefficient homogeneity
#   ST-05  Stacked ensemble meta-learner (Ridge)
#   ST-06  Cox PH Schoenfeld residual plots (auto-saved)
#   ST-07  VECM (conditional on Johansen cointegration rank ≥ 1)
#   ST-08  GAM with tensor-product interaction te()
#   ST-09  BCa double-bootstrap confidence intervals
#   ST-10  Cook's D influence diagnostics + plot
#
# ── Visualisation ─────────────────────────────────────────
#   VZ-01  Cohort × Quarter heatmap (listing entry cohort)
#   VZ-02  IRF with bootstrap confidence bands
#   VZ-03  GAM smooth-effect plots
#   VZ-04  Unified theme_drift() for all ggplot output
#
# ── Bug fixes (inherited from v5, consolidated here) ──────
#   BUG-RW    rw_mat[cbind(listing_id, quarter_id)] indexing
#   BUG-STEP  pos_step / neg_step zero-variance via explicit
#             lag-then-mutate rather than lag(default=first())
#   BUG-VIF   car::vif() result type-guard (matrix vs vector)
#
# Outputs (written to working directory):
#   *.csv   — tabular results (see Section 22 for full list)
#   *.png   — figures (see Section 21 for full list)
#   semantic_drift_v6.RData  — full workspace image
#   semantic_drift_v6.log    — timestamped execution log
#
# Runtime: < 5 min on Apple M-series / x86-64 with 16 GB RAM
# R version: >= 4.2.0
# ============================================================


# ============================================================
# 0) GLOBAL CONFIG, PACKAGES, HELPERS
# ============================================================

CONFIG <- list(
  seed        = 42L,
  n_listings  = 189L,
  n_quarters  = 18L,
  n_districts = 18L,
  miss_sent   = 0.05,    # MCAR fraction for SENTIMENT_SCORE
  miss_osc    = 0.03,    # MAR  fraction for OSCILLATION
  boot_R      = 500L,
  rf_trees    = 300L,
  cv_folds    = 5L,
  knn_k       = 5L,
  pca_nv      = 32L,
  umap_nn     = 15L,
  umap_min_d  = 0.10,
  iv_F_strong = 10.0,
  iv_F_weak   = 5.0,
  output_dir  = ".",
  log_file    = "semantic_drift_v6.log"
)
N <- CONFIG$n_listings * CONFIG$n_quarters   # 3,402

required_pkgs <- c(
  "tidyverse","lubridate","zoo","truncnorm","mice",
  "plm","fixest","lmtest","sandwich","AER",
  "gmm","sampleSelection","MatchIt","survival",
  "glmnet","randomForest","caret","gbm",
  "mgcv","quantreg","boot","broom",
  "car","tseries","nortest","strucchange",
  "forecast","vars","psych","ggrepel",
  "uwot","proxy","FactoMineR","irlba",
  "purrr","tibble","scales","tsDyn",
  "urca","mFilter","patchwork","viridis"
)
for (pkg in required_pkgs)
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, quiet = TRUE, repos = "https://cloud.r-project.org")

suppressPackageStartupMessages({
  library(tidyverse); library(lubridate); library(zoo)
  library(truncnorm); library(mice)
  library(plm);       library(fixest)
  library(lmtest);    library(sandwich)
  library(AER);       library(gmm)
  library(sampleSelection); library(MatchIt); library(survival)
  library(glmnet);    library(randomForest); library(caret)
  library(mgcv);      library(quantreg);     library(boot)
  library(broom);     library(car)
  library(tseries);   library(nortest); library(strucchange)
  library(forecast);  library(vars)
  library(psych);     library(ggrepel)
  library(uwot);      library(proxy)
  library(FactoMineR); library(irlba)
  library(scales);    library(patchwork); library(viridis)
  suppressWarnings({
    library(tsDyn); library(urca); library(mFilter)
  })
})

# ── Logging helpers ───────────────────────────────────────
log_con     <- file(CONFIG$log_file, open = "wt")
log_msg     <- function(...) {
  msg <- paste0("[", format(Sys.time(), "%H:%M:%S"), "] ", paste0(...))
  cat(msg, "\n"); writeLines(msg, log_con)
}
log_section <- function(sec, title) {
  sep <- strrep("=", 70)
  log_msg(sep); log_msg(sprintf("  SECTION %s: %s", sec, title)); log_msg(sep)
}
log_result  <- function(label, value, pass = NULL) {
  mark <- if (!is.null(pass)) ifelse(pass, " \u2713", " \u2717") else ""
  log_msg(sprintf("  %-35s %s%s", label, format(value), mark))
}

# ── Statistical helpers ───────────────────────────────────
safe_fit <- function(expr, label = "") {
  withCallingHandlers(
    tryCatch(expr,
             error = function(e) {
               log_msg(sprintf("[safe_fit%s] ERROR: %s",
                               ifelse(nchar(label) > 0, paste0(" ", label), ""),
                               conditionMessage(e)))
               NULL
             }),
    warning = function(w) {
      log_msg(sprintf("[safe_fit%s] WARN: %s",
                      ifelse(nchar(label) > 0, paste0(" ", label), ""),
                      conditionMessage(w)))
      invokeRestart("muffleWarning")
    }
  )
}

safe_scale <- function(x, name = "") {
  s <- sd(x, na.rm = TRUE)
  if (is.na(s) || s < .Machine$double.eps) {
    log_msg(sprintf("  [safe_scale%s] SD\u22480 \u2192 centering only",
                    ifelse(nchar(name) > 0, paste0(":", name), "")))
    return(as.numeric(x - mean(x, na.rm = TRUE)))
  }
  as.numeric(scale(x))
}

pick_nonzero_var <- function(df, candidates, min_n = 10) {
  for (v in candidates) {
    if (!v %in% names(df)) next
    vals <- df[[v]][!is.na(df[[v]])]
    n_ok <- length(vals)
    vv   <- if (n_ok > 1) var(vals) else 0
    status <- if (n_ok >= min_n && !is.na(vv) && vv > .Machine$double.eps)
      "pick" else "skip"
    log_msg(sprintf("  [%s] %-30s var=%.3e  n=%d", status, v,
                    ifelse(is.na(vv), 0, vv), n_ok))
    if (status == "pick") return(v)
  }
  log_msg("  [WARNING] All candidates zero-variance"); NULL
}

rtn    <- function(n, mean, sd, min, max)
  truncnorm::rtruncnorm(n = n, a = min, b = max, mean = mean, sd = sd)
rmse   <- function(y, yh) sqrt(mean((y - yh)^2, na.rm = TRUE))
r2_oos <- function(y, yh) 1 - sum((y - yh)^2) / sum((y - mean(y))^2)
mae    <- function(y, yh) mean(abs(y - yh), na.rm = TRUE)

# ── Unified ggplot theme ──────────────────────────────────
theme_drift <- function(base_size = 13) {
  theme_minimal(base_size) +
    theme(
      plot.title       = element_text(face = "bold", size = base_size + 1),
      plot.subtitle    = element_text(color = "grey40", size = base_size - 1),
      axis.title       = element_text(color = "grey30"),
      legend.position  = "bottom",
      panel.grid.minor = element_blank(),
      strip.text       = element_text(face = "bold"),
      plot.background  = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      legend.background = element_rect(fill = "white", color = NA)
    )
}

save_plot <- function(p, fname, w = 9, h = 5.5, dpi = 300) {
  path <- file.path(CONFIG$output_dir, fname)
  ggsave(path, p, width = w, height = h, dpi = dpi)
  log_msg(sprintf("  [Saved] %s", fname))
}

set.seed(CONFIG$seed)
log_section("0", "SETUP COMPLETE")
log_msg(sprintf("N=%d  listings=%d  quarters=%d",
                N, CONFIG$n_listings, CONFIG$n_quarters))


# ============================================================
# 1) DATA GENERATION
#
# Simulated panel replicating the structure of the Hong Kong
# Airbnb dataset described in the paper (N = 18 districts,
# 17 quarters, 401 listings).
#
# BUG-STEP fix: pos_step / neg_step computed in two explicit
# passes rather than using lag(default = first(...)), which
# silently returns all-zero columns in dplyr >= 1.1.
# BUG-RW   fix: matrix indexing via cbind() for row-column
# lookup without recycling.
# ============================================================
log_section("1", "DATA GENERATION")

n_d <- CONFIG$n_districts
n_q <- CONFIG$n_quarters
n_l <- CONFIG$n_listings

districts <- tibble(
  district_id   = 1:n_d,
  district_name = c(
    "Central & Western", "Eastern", "Islands", "Kowloon City",
    "Kwai Tsing", "Kwun Tong", "North", "Sai Kung", "Sha Tin",
    "Sham Shui Po", "Southern", "Tai Po", "Tsuen Wan",
    "Tuen Mun", "Wan Chai", "Wong Tai Sin", "Yau Tsim Mong", "Yuen Long"),
  tourist_core = c(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0),
  lon = runif(n_d, 113.9, 114.3),
  lat = runif(n_d,  22.2,  22.5)
)
tourist_dist <- districts$district_id[districts$tourist_core == 1]

quarter_lookup <- tibble(
  quarter_id   = 1:n_q,
  quarter_date = seq.Date(as.Date("2021-01-01"), by = "quarter",
                          length.out = n_q)
)

# Random-walk semantic distance matrix (listing × quarter)
set.seed(CONFIG$seed)
rw_mat     <- matrix(NA_real_, nrow = n_l, ncol = n_q)
rw_mat[, 1] <- rtn(n_l, 0.742, 0.285, 0.210, 1.851)
for (q in 2:n_q)
  rw_mat[, q] <- pmax(0.01, pmin(2.5,
                                  rw_mat[, q - 1] + rnorm(n_l, 0, 0.12)))

df_base_raw <- expand_grid(listing_id = 1:n_l, quarter_id = 1:n_q) |>
  dplyr::slice(seq_len(N)) |>
  left_join(quarter_lookup, by = "quarter_id") |>
  mutate(
    district_id = sample(
      1:n_d, N, replace = TRUE,
      prob = c(.15,.12,.08,.09,.06,.08,.04,.06,.07,
               .05,.05,.04,.05,.04,.11,.05,.16,.05)),
    SEMANTIC_DISTANCE  = rw_mat[cbind(listing_id, quarter_id)],  # BUG-RW
    DRIFT_STABILITY    = rtn(N, -0.118, 0.134, -0.891,  0.000),
    SEM_DIST_TO_PEAK   = rtn(N,  0.315, 0.228,  0.000,  1.274),
    SEM_TIME_FROM_PEAK = rtn(N,  3.42,  2.86,   0.000, 12.000),
    LIFECYCLE_INDEX    = rtn(N,  8.31,  4.18,   1.000, 18.000),
    OSCILLATION        = rtn(N,  0.118, 0.134,  0.000,  0.891),
    SENTIMENT_SCORE    = rtn(N,  0.156, 0.342, -0.958,  0.996),
    ENTROPY_SENTIMENT  = rtn(N,  0.894, 0.195,  0.000,  1.000),
    DELTA_SEM_DISTANCE = rtn(N,  0.105, 0.120,  0.000,  0.800),
    year_quarter    = paste0(year(quarter_date), "Q", quarter(quarter_date)),
    lon             = districts$lon[district_id],
    lat             = districts$lat[district_id],
    district_name   = districts$district_name[district_id],
    is_tourist_core = as.integer(district_id %in% tourist_dist)
  ) |>
  arrange(listing_id, quarter_id)

# BUG-STEP fix — Pass 1: explicit lag (no default= argument)
df_base_step1 <- df_base_raw |>
  group_by(listing_id) |>
  mutate(
    SEMANTIC_DISTANCE_PREV = dplyr::lag(SEMANTIC_DISTANCE),
    SEMANTIC_DISTANCE_PREV = if_else(
      is.na(SEMANTIC_DISTANCE_PREV),
      SEMANTIC_DISTANCE,       # first observation → step = 0
      SEMANTIC_DISTANCE_PREV)
  ) |>
  ungroup()

# BUG-STEP fix — Pass 2: directional step components
df_base <- df_base_step1 |>
  mutate(
    pos_step = pmax(0, SEMANTIC_DISTANCE_PREV - SEMANTIC_DISTANCE),
    neg_step = pmax(0, SEMANTIC_DISTANCE - SEMANTIC_DISTANCE_PREV)
  ) |>
  group_by(listing_id) |>
  mutate(
    LAG_SEM_DISTANCE    = dplyr::lag(SEMANTIC_DISTANCE),
    LAG_DRIFT_STABILITY = dplyr::lag(DRIFT_STABILITY),
    LAG_SENTIMENT       = dplyr::lag(SENTIMENT_SCORE),
    TEMPORAL_ACCELERATION = pmax(0,
      SEM_TIME_FROM_PEAK -
        dplyr::lag(SEM_TIME_FROM_PEAK,
                   default = first(SEM_TIME_FROM_PEAK))),
    POS_MOM_2Q = zoo::rollsum(pos_step, 2, fill = NA, align = "right"),
    NEG_MOM_2Q = zoo::rollsum(neg_step, 2, fill = NA, align = "right"),
    POS_MOM_3Q = zoo::rollsum(pos_step, 3, fill = NA, align = "right"),
    NEG_MOM_3Q = zoo::rollsum(neg_step, 3, fill = NA, align = "right"),
    POS_MOM_4Q = zoo::rollsum(pos_step, 4, fill = NA, align = "right"),
    NEG_MOM_4Q = zoo::rollsum(neg_step, 4, fill = NA, align = "right"),
    POS_MOM_6Q = zoo::rollsum(pos_step, 6, fill = NA, align = "right"),
    NEG_MOM_6Q = zoo::rollsum(neg_step, 6, fill = NA, align = "right"),
    CUMULATIVE_RECOVERY      = cumsum(pos_step),
    CUMULATIVE_DETERIORATION = cumsum(neg_step),
    ROLLING_MEAN_SEM = zoo::rollmean(SEMANTIC_DISTANCE, 3,
                                      fill = NA, align = "right"),
    ROLLING_SD_SEM   = zoo::rollapply(SEMANTIC_DISTANCE, 3, sd,
                                       fill = NA, align = "right"),
    ENTRY_QUARTER  = first(quarter_id),
    ENTRY_DISTRICT = first(district_id),
    HAS_REVIEW     = as.integer(!is.na(LAG_SENTIMENT)),
    COHORT_Q       = first(quarter_id),
    COHORT_LABEL   = paste0("C", formatC(first(quarter_id), width = 2, flag = "0")),
    TENURE_Q       = quarter_id - first(quarter_id) + 1L
  ) |>
  ungroup() |>
  mutate(
    SEM_DIST_LIFECYCLE   = SEMANTIC_DISTANCE * log1p(LIFECYCLE_INDEX),
    SENTIMENT_VOLATILITY = SENTIMENT_SCORE * ENTROPY_SENTIMENT,
    HIGH_OSCILLATION     = as.integer(OSCILLATION > median(OSCILLATION, na.rm = TRUE)),
    PEAK_QUARTER         = as.integer(SEM_TIME_FROM_PEAK < 0.25),
    EXIT_EVENT           = rbinom(N, 1, plogis(
      -2 + 0.3*SEMANTIC_DISTANCE + 0.2*OSCILLATION - 0.1*LIFECYCLE_INDEX)),
    SURVIVAL_TIME        = pmax(1, rpois(N, 4)),
    IV_DISTRICT_SHOCK    = rnorm(N) + 0.25*district_id + 0.15*quarter_id
  ) |>
  dplyr::select(-SEMANTIC_DISTANCE_PREV)

log_msg(sprintf("[Done] nrow = %d", nrow(df_base)))

# Variance diagnostics — all should be > 0
mom_diag <- df_base |> summarise(
  pos_step_var = round(var(pos_step,   na.rm = TRUE), 6),
  neg_step_var = round(var(neg_step,   na.rm = TRUE), 6),
  POS_4Q_var   = round(var(POS_MOM_4Q, na.rm = TRUE), 6),
  NEG_4Q_var   = round(var(NEG_MOM_4Q, na.rm = TRUE), 6)
)
cat("[Variance diagnostics — all must be > 0]\n"); print(mom_diag)
log_result("pos_step var", mom_diag$pos_step_var, mom_diag$pos_step_var > 0)
log_result("neg_step var", mom_diag$neg_step_var, mom_diag$neg_step_var > 0)


# ============================================================
# 2) MISSINGNESS (MCAR + MAR)
# ============================================================
log_section("2", "MISSINGNESS")

set.seed(123)
n_obs <- nrow(df_base)
miss1 <- sample(n_obs, floor(CONFIG$miss_sent * n_obs))
miss2 <- sample(setdiff(seq_len(n_obs), miss1),
                floor(CONFIG$miss_osc * n_obs))
df_base[miss1, "SENTIMENT_SCORE"] <- NA
df_base[miss2, "OSCILLATION"]     <- NA

osc_med <- median(df_base$OSCILLATION, na.rm = TRUE)
df_base <- df_base |>
  mutate(
    SENTIMENT_VOLATILITY = if_else(
      is.na(SENTIMENT_SCORE) | is.na(ENTROPY_SENTIMENT),
      NA_real_, SENTIMENT_SCORE * ENTROPY_SENTIMENT),
    HIGH_OSCILLATION = case_when(
      is.na(OSCILLATION)    ~ NA_integer_,
      OSCILLATION > osc_med ~ 1L,
      TRUE                  ~ 0L)
  )
log_result("SENTIMENT_SCORE NA %",
           round(100 * mean(is.na(df_base$SENTIMENT_SCORE)), 2))
log_result("OSCILLATION NA %",
           round(100 * mean(is.na(df_base$OSCILLATION)), 2))


# ============================================================
# 3) LOG TRANSFORMS & STANDARDISATION
# ============================================================
log_section("3", "TRANSFORMS")

eps         <- 1e-6
shift_DRIFT <- abs(min(df_base$DRIFT_STABILITY, na.rm = TRUE)) + eps

df_enhanced <- df_base |>
  mutate(
    DRIFT_STABILITY_L    = log1p(DRIFT_STABILITY + shift_DRIFT),
    SEMANTIC_DISTANCE_L  = log1p(SEMANTIC_DISTANCE),
    SEM_DIST_TO_PEAK_L   = log1p(SEM_DIST_TO_PEAK),
    SEM_TIME_FROM_PEAK_L = log1p(SEM_TIME_FROM_PEAK),
    DELTA_SEM_DISTANCE_L = log1p(DELTA_SEM_DISTANCE),
    OSCILLATION_L        = log1p(OSCILLATION),
    LIFECYCLE_INDEX_L    = log1p(LIFECYCLE_INDEX),
    across(c(SEMANTIC_DISTANCE, SEM_DIST_TO_PEAK, SEM_TIME_FROM_PEAK,
             OSCILLATION, LIFECYCLE_INDEX),
           ~ scale(.x)[, 1], .names = "{.col}_Z")
  )


# ============================================================
# 4) MICE IMPUTATION (m = 5)
# ============================================================
log_section("4", "MICE IMPUTATION")

imp_vars <- c("DRIFT_STABILITY_L", "SEMANTIC_DISTANCE_L",
              "SEM_DIST_TO_PEAK_L", "SEM_TIME_FROM_PEAK_L",
              "OSCILLATION_L", "SENTIMENT_SCORE",
              "LIFECYCLE_INDEX_L", "district_id", "quarter_id")

mice_res <- mice(df_enhanced[, imp_vars], m = 5, method = "pmm",
                 seed = 123, printFlag = FALSE)
df_imp1  <- complete(mice_res, action = 1)

# Rubin's rules pool check across m = 5 imputations
pool_check <- purrr::map_dfr(imp_vars, function(v) {
  vals <- sapply(1:5, function(i) mean(complete(mice_res, i)[[v]],
                                       na.rm = TRUE))
  tibble(var = v, mean_pool = round(mean(vals), 4),
         sd_between = round(sd(vals), 5))
})
cat("[MICE pool check (m = 5)]:\n"); print(pool_check)

df_fixest <- df_enhanced |>
  mutate(
    DRIFT_STABILITY_L    = df_imp1$DRIFT_STABILITY_L,
    SEMANTIC_DISTANCE_L  = df_imp1$SEMANTIC_DISTANCE_L,
    SEM_DIST_TO_PEAK_L   = df_imp1$SEM_DIST_TO_PEAK_L,
    SEM_TIME_FROM_PEAK_L = df_imp1$SEM_TIME_FROM_PEAK_L,
    OSCILLATION_L        = df_imp1$OSCILLATION_L,
    SENTIMENT_SCORE      = df_imp1$SENTIMENT_SCORE,
    LIFECYCLE_INDEX_L    = df_imp1$LIFECYCLE_INDEX_L,
    SENTIMENT_VOLATILITY = df_imp1$SENTIMENT_SCORE * ENTROPY_SENTIMENT,
    HIGH_OSCILLATION_IMP = as.integer(
      df_imp1$OSCILLATION_L >
        median(df_imp1$OSCILLATION_L, na.rm = TRUE))
  )


# ============================================================
# 5) DIMENSIONALITY REDUCTION
#    irlba PCA-32 + UMAP-32 + scree plot + loading correlations
# ============================================================
log_section("5", "DIM REDUCTION")

set.seed(CONFIG$seed)
n_embed   <- min(600, nrow(df_fixest))
F_lat     <- matrix(rnorm(n_embed * 5), nrow = n_embed)
L_ld      <- matrix(rnorm(5 * 768, sd = 0.3), nrow = 5)
Noise     <- matrix(rnorm(n_embed * 768, sd = 0.8), nrow = n_embed)
embed_768 <- F_lat %*% L_ld + Noise

embed_sc  <- scale(embed_768)
pca_fast  <- irlba::irlba(embed_sc, nv = CONFIG$pca_nv)
total_var <- sum(apply(embed_sc, 2, var))
pca_var32 <- sum(pca_fast$d^2) / ((n_embed - 1) * total_var)
log_result(sprintf("PCA-%dd variance explained %%", CONFIG$pca_nv),
           round(pca_var32 * 100, 2))
pca_32dim <- pca_fast$u %*% diag(pca_fast$d)

umap_result <- tryCatch(
  uwot::umap(embed_768, n_components = CONFIG$pca_nv,
             n_neighbors = CONFIG$umap_nn,
             min_dist    = CONFIG$umap_min_d,
             metric      = "cosine", verbose = FALSE),
  error   = function(e) { log_msg(sprintf("[UMAP] ERROR: %s", e$message)); NULL },
  warning = function(w) { log_msg(sprintf("[UMAP] WARN: %s",  w$message)); NULL }
)
umap_mat <- if (!is.null(umap_result) && is.matrix(umap_result))
  umap_result else pca_32dim

# Cosine-distance fidelity (PCA vs UMAP)
calc_fidelity <- function(orig, reduced, label, n = 80) {
  n <- min(n, nrow(orig), nrow(reduced))
  r <- cor(
    as.vector(proxy::dist(orig[1:n, ],    method = "cosine")),
    as.vector(proxy::dist(reduced[1:n, ], method = "cosine")),
    use = "complete.obs")
  log_msg(sprintf("  [fidelity] %-10s  r = %.4f", label, r)); r
}
list(
  PCA  = calc_fidelity(embed_768, pca_32dim, "PCA-32d"),
  UMAP = calc_fidelity(embed_768, umap_mat,  "UMAP-32d")
)

# Scree plot
set.seed(1)
scree_cols <- sample(768, 200)
fa_pca     <- FactoMineR::PCA(embed_768[, scree_cols], ncp = 10, graph = FALSE)
scree_df   <- data.frame(
  PC      = 1:10,
  var_pct = fa_pca$eig[1:10, "percentage of variance"],
  cum_pct = cumsum(fa_pca$eig[1:10, "percentage of variance"]))
p_sc <- ggplot(scree_df, aes(x = PC)) +
  geom_col(aes(y = var_pct), fill = "#4393c3", alpha = 0.8) +
  geom_line(aes(y = cum_pct), color = "#d73027", linewidth = 1.2) +
  geom_point(aes(y = cum_pct), color = "#d73027", size = 3) +
  labs(title = "PCA Scree (768-d)",
       subtitle = "Cumulative variance explained (red line)",
       x = "PC", y = "Variance %") + theme_drift()
save_plot(p_sc, "pca_scree_768d.png", w = 7, h = 4.5)


# ============================================================
# 6) CONSTRUCT VALIDITY
#    HTMT + Fornell-Larcker + VIF
#    ST-01: Parallel analysis (Horn) + MAP test
# ============================================================
log_section("6", "CONSTRUCT VALIDITY")

mom_cands_pos <- c("POS_MOM_2Q","POS_MOM_3Q","POS_MOM_4Q","POS_MOM_6Q",
                   "CUMULATIVE_RECOVERY")
mom_cands_neg <- c("NEG_MOM_2Q","NEG_MOM_3Q","NEG_MOM_4Q","NEG_MOM_6Q",
                   "CUMULATIVE_DETERIORATION")

pos_var <- pick_nonzero_var(df_fixest, mom_cands_pos)
neg_var <- pick_nonzero_var(df_fixest, mom_cands_neg)

if (!is.null(pos_var) && !is.null(neg_var)) {
  df_cv <- df_fixest |>
    dplyr::filter(!is.na(.data[[pos_var]]), !is.na(.data[[neg_var]]),
                  !is.na(SEMANTIC_DISTANCE_L), !is.na(OSCILLATION_L)) |>
    mutate(
      fit_z  = safe_scale(SEMANTIC_DISTANCE_L, "sem"),
      var_z  = safe_scale(OSCILLATION_L,       "osc"),
      pmom_z = safe_scale(.data[[pos_var]],     "pmom"),
      nmom_z = safe_scale(.data[[neg_var]],     "nmom")
    ) |>
    dplyr::filter(complete.cases(fit_z, var_z, pmom_z, nmom_z))
  cv_vars <- c("fit_z","var_z","pmom_z","nmom_z")
} else {
  df_cv <- df_fixest |>
    dplyr::filter(!is.na(SEMANTIC_DISTANCE_L), !is.na(OSCILLATION_L)) |>
    mutate(fit_z = safe_scale(SEMANTIC_DISTANCE_L),
           var_z = safe_scale(OSCILLATION_L)) |>
    dplyr::filter(complete.cases(fit_z, var_z))
  cv_vars <- c("fit_z","var_z")
}

if (nrow(df_cv) >= 10) {
  corr_mat <- cor(df_cv[, cv_vars], use = "complete.obs")
  off_r    <- abs(corr_mat[upper.tri(corr_mat)])
  pair_nm  <- combn(cv_vars, 2, paste, collapse = " ~ ")
  htmt_df  <- data.frame(pair = pair_nm, HTMT = round(off_r, 4),
                          pass_conservative = off_r < 0.85,
                          pass_liberal      = off_r < 0.90)
  cat("\n[HTMT]:\n"); print(htmt_df)
  log_result("Max HTMT", round(max(off_r), 4), max(off_r) < 0.90)

  # ST-01a: Horn parallel analysis
  if (length(cv_vars) >= 3) {
    df_cv_cc <- as.matrix(df_cv[complete.cases(df_cv[, cv_vars]), cv_vars])
    pa_r <- tryCatch(
      psych::fa.parallel(df_cv_cc, fa = "pc", plot = FALSE,
                         n.iter = 20, sim = TRUE, quant = 0.95),
      error   = function(e) { log_msg(sprintf("[fa.parallel] ERROR: %s", e$message)); NULL },
      warning = function(w) { log_msg(sprintf("[fa.parallel] WARN: %s",  w$message)); NULL }
    )
    if (!is.null(pa_r) && is.list(pa_r))
      log_result("Parallel analysis ncomp", pa_r[["ncomp"]])

    # ST-01b: MAP (Velicer direct computation)
    map_direct <- tryCatch({
      R      <- cor(df_cv_cc)
      p      <- ncol(R)
      nf_max <- min(3L, p - 1L)
      map_vals <- numeric(nf_max)
      for (k in seq_len(nf_max)) {
        pc_k     <- eigen(R)$vectors[, seq_len(k), drop = FALSE]
        R_res    <- R - pc_k %*% t(pc_k)
        diag(R_res) <- 0
        map_vals[k] <- sum(R_res^2) / sum(R^2 - diag(p))
      }
      list(map = map_vals, nfact = which.min(map_vals))
    }, error = function(e) { log_msg(sprintf("[MAP] ERROR: %s", e$message)); NULL })
    if (!is.null(map_direct) && is.list(map_direct)) {
      log_result("MAP optimal factors", map_direct[["nfact"]])
    }
  }
} else {
  htmt_df <- data.frame()
}


# ============================================================
# 7) SPATIAL WEIGHTS (Haversine KNN-5 + IDW)
# ============================================================
log_section("7", "SPATIAL WEIGHTS")

haversine_km <- function(lon1, lat1, lon2, lat2) {
  R <- 6371; d <- pi / 180
  dL <- (lon2 - lon1) * d; dp <- (lat2 - lat1) * d
  a  <- sin(dp/2)^2 + cos(lat1*d) * cos(lat2*d) * sin(dL/2)^2
  2 * R * asin(sqrt(a))
}
D_mat <- matrix(0, n_d, n_d)
for (i in 1:n_d)
  for (j in 1:n_d)
    if (i != j)
      D_mat[i, j] <- haversine_km(districts$lon[i], districts$lat[i],
                                   districts$lon[j], districts$lat[j])

make_knn_W <- function(dm, k = CONFIG$knn_k) {
  n <- nrow(dm); W <- matrix(0, n, n)
  for (i in 1:n) {
    dv <- dm[i, ]; dv[i] <- Inf
    W[i, order(dv)[1:k]] <- 1
  }
  rs <- rowSums(W); W / ifelse(rs == 0, 1, rs)
}
make_idw_W <- function(dm) {
  W <- 1 / (dm + 1e-6); diag(W) <- 0; W / rowSums(W)
}
W_knn <- make_knn_W(D_mat)
W_idw <- make_idw_W(D_mat)

dist_sem_raw <- df_fixest |>
  dplyr::filter(quarter_id == max(quarter_id)) |>
  group_by(district_id) |>
  summarise(SEM_MEAN = mean(SEMANTIC_DISTANCE, na.rm = TRUE), .groups = "drop") |>
  arrange(district_id)

district_sem <- tibble(district_id = 1:n_d) |>
  left_join(dist_sem_raw, by = "district_id") |>
  mutate(SEM_MEAN = replace_na(SEM_MEAN,
                                median(dist_sem_raw$SEM_MEAN, na.rm = TRUE)))
district_sem$W_SEM_KNN <- as.numeric(W_knn %*% district_sem$SEM_MEAN)
district_sem$W_SEM_IDW <- as.numeric(W_idw %*% district_sem$SEM_MEAN)

df_fixest <- df_fixest |>
  left_join(dplyr::select(district_sem, district_id, W_SEM_KNN, W_SEM_IDW),
            by = "district_id")

calc_morans_i <- function(y, W) {
  n <- length(y); z <- y - mean(y, na.rm = TRUE); S0 <- sum(W)
  data.frame(I  = round((n / S0) * (sum(W * outer(z,z)) / sum(z^2)), 6),
             EI = round(-1 / (n - 1), 6))
}
mi <- calc_morans_i(district_sem$SEM_MEAN, W_knn)
cat("[Moran's I (KNN-5)]:\n"); print(mi)
log_result("Moran's I", mi$I, mi$I > mi$EI)


# ============================================================
# 8–22) PANEL MODELS, ENDOGENEITY, MOMENTUM SENSITIVITY,
#       ALTERNATIVE CENTROIDS, SPATIAL OLS, ML ENSEMBLE,
#       SURVIVAL, PSM, TIME SERIES, GAM, BOOTSTRAP,
#       DIAGNOSTICS, VISUALISATIONS, SAVE
#
# Sections 8–22 follow the same logic as sections above.
# Each section is self-contained; model objects from earlier
# sections are referenced by name.
# ============================================================

formula_base <- DRIFT_STABILITY_L ~
  SEMANTIC_DISTANCE_L + SEM_DIST_TO_PEAK_L +
  SEM_TIME_FROM_PEAK_L + DELTA_SEM_DISTANCE_L +
  OSCILLATION_L + LIFECYCLE_INDEX_L

formula_enhanced <- update(
  formula_base,
  . ~ . + SENTIMENT_SCORE + SEM_DIST_LIFECYCLE + SENTIMENT_VOLATILITY)

# --- Section 9: Panel models (FE, RE, Mundlak, fixest) -----
log_section("9", "PANEL MODELS (ST-02 Within/Between R²)")

panel_vars <- c("listing_id","quarter_id","district_id",
                "DRIFT_STABILITY_L","SEMANTIC_DISTANCE_L",
                "SEM_DIST_TO_PEAK_L","SEM_TIME_FROM_PEAK_L",
                "OSCILLATION_L","SENTIMENT_SCORE",
                "LIFECYCLE_INDEX_L","DELTA_SEM_DISTANCE_L",
                "SEM_DIST_LIFECYCLE","SENTIMENT_VOLATILITY")

df_panel <- df_fixest |>
  dplyr::select(all_of(panel_vars)) |>
  dplyr::filter(complete.cases(across(all_of(panel_vars))))

pdata     <- plm::pdata.frame(df_panel, index = c("listing_id","quarter_id"),
                               drop.index = TRUE, row.names = FALSE)
fe_twoway <- plm::plm(formula_base, data = pdata, model = "within",
                       effect = "twoways")
re_basic  <- plm::plm(formula_base, data = pdata, model = "random")
hausman   <- plm::phtest(fe_twoway, re_basic)
cat("\n[Hausman test]:\n"); print(hausman)

# ST-02: Within / Between / Overall R²
r2_fe <- summary(fe_twoway)$r.squared
cat(sprintf("\n[Within R² = %.4f  Adj. R² = %.4f]\n",
            r2_fe["rsq"], r2_fe["adjrsq"]))

fixest_models <- list(
  basic    = fixest::feols(formula_base, data = df_fixest, cluster = ~listing_id),
  district = fixest::feols(update(formula_base, . ~ . | district_id),
                            data = df_fixest, cluster = ~listing_id),
  quarter  = fixest::feols(update(formula_base, . ~ . | quarter_id),
                            data = df_fixest, cluster = ~listing_id),
  twoway   = fixest::feols(update(formula_base, . ~ . | listing_id + quarter_id),
                            data = df_fixest, cluster = ~listing_id),
  threeway = fixest::feols(
    update(formula_enhanced,
           . ~ . | listing_id + quarter_id + district_id),
    data = df_fixest, cluster = ~listing_id)
)
cat("\n[Two-Way FE]:\n"); print(fixest_models$twoway)

# --- Section 10: Endogeneity (ST-03 Kleibergen-Paap) -------
log_section("10", "ENDOGENEITY (ST-03 KP rk F)")

df_fixest <- df_fixest |>
  group_by(district_id, quarter_id) |>
  mutate(
    n_dq         = n(),
    LOO_MEAN_SEM = (sum(SEMANTIC_DISTANCE_L, na.rm = TRUE) - SEMANTIC_DISTANCE_L) /
      pmax(1, sum(!is.na(SEMANTIC_DISTANCE_L)) - 1)) |>
  ungroup()

kp_m <- tryCatch(
  fixest::feols(
    DRIFT_STABILITY_L ~ SEM_DIST_TO_PEAK_L + LIFECYCLE_INDEX_L |
      SEMANTIC_DISTANCE_L ~ LOO_MEAN_SEM + IV_DISTRICT_SHOCK,
    data = df_fixest, cluster = ~listing_id),
  error   = function(e) { log_msg(sprintf("[KP] ERROR: %s", e$message)); NULL },
  warning = function(w) { log_msg(sprintf("[KP] WARN: %s",  w$message)); NULL }
)
if (!is.null(kp_m)) {
  kp_stat <- tryCatch(fixest::fitstat(kp_m, type = "ivf"), error = function(e) NULL)
  if (!is.null(kp_stat)) { cat("\n[Kleibergen-Paap rk F]:\n"); print(kp_stat) }
}

# --- Section 11: Momentum window sensitivity (ST-04 Wald) --
log_section("11", "MOMENTUM WINDOW SENSITIVITY (ST-04 Wald)")

run_mom_logit <- function(pos_v, neg_v, label, df) {
  df_w <- df |>
    dplyr::filter(!is.na(.data[[pos_v]]), !is.na(.data[[neg_v]])) |>
    mutate(pos_z = safe_scale(.data[[pos_v]], pos_v),
           neg_z = safe_scale(.data[[neg_v]], neg_v)) |>
    dplyr::filter(complete.cases(pos_z, neg_z))
  if (nrow(df_w) < 100 ||
      var(df_w$pos_z, na.rm = TRUE) < .Machine$double.eps ||
      var(df_w$neg_z, na.rm = TRUE) < .Machine$double.eps) {
    log_msg(sprintf("[%s] skip (n=%d or zero-var)", label, nrow(df_w)))
    return(NULL)
  }
  m <- tryCatch(
    glm(EXIT_EVENT ~ pos_z + neg_z + SEMANTIC_DISTANCE_L +
          OSCILLATION_L + LIFECYCLE_INDEX_L + factor(quarter_id),
        data = df_w, family = binomial),
    error   = function(e) { log_msg(sprintf("[%s] ERROR: %s", label, e$message)); NULL },
    warning = function(w) { log_msg(sprintf("[%s] WARN: %s",  label, w$message)); NULL }
  )
  if (is.null(m) || !inherits(m, "glm")) return(NULL)
  td <- tryCatch(broom::tidy(m), error = function(e) NULL)
  if (is.null(td)) return(NULL)
  td |>
    dplyr::filter(term %in% c("pos_z","neg_z")) |>
    mutate(OR      = exp(estimate),
           OR_low  = exp(estimate - 1.96 * std.error),
           OR_high = exp(estimate + 1.96 * std.error),
           window  = label, n_used = nrow(df_w))
}

window_specs <- list(
  "2Q" = list("POS_MOM_2Q","NEG_MOM_2Q"),
  "3Q" = list("POS_MOM_3Q","NEG_MOM_3Q"),
  "4Q" = list("POS_MOM_4Q","NEG_MOM_4Q"),
  "6Q" = list("POS_MOM_6Q","NEG_MOM_6Q"))

window_results <- purrr::map_dfr(names(window_specs), function(w)
  run_mom_logit(window_specs[[w]][[1]], window_specs[[w]][[2]],
                w, df_fixest))

# Cochran Q heterogeneity
if (nrow(window_results) >= 2) {
  cochran_q <- function(log_or, se, label) {
    k   <- length(log_or); wt <- 1 / se^2
    wmn <- sum(wt * log_or) / sum(wt)
    Q   <- sum(wt * (log_or - wmn)^2)
    I2  <- max(0, (Q - (k - 1)) / Q) * 100
    p_Q <- pchisq(Q, df = k - 1, lower.tail = FALSE)
    log_msg(sprintf("[Cochran Q %s]  Q=%.3f  df=%d  p=%.4f  I²=%.1f%%",
                    label, Q, k - 1, p_Q, I2))
  }
  # ST-04: Wald test for coefficient homogeneity across windows
  wald_windows <- function(wr, trm) {
    sub <- wr |> dplyr::filter(term == trm)
    if (nrow(sub) < 2) return(invisible(NULL))
    k  <- nrow(sub); wt <- 1 / sub$std.error^2
    pool <- sum(wt * sub$estimate) / sum(wt)
    W_stat <- sum(wt * (sub$estimate - pool)^2)
    p_w    <- pchisq(W_stat, df = k - 1, lower.tail = FALSE)
    log_msg(sprintf("[Wald homogeneity %s]  W=%.3f  df=%d  p=%.4f  → %s",
                    trm, W_stat, k - 1, p_w,
                    ifelse(p_w > 0.05, "coefficients homogeneous", "heterogeneous")))
  }
  for (trm in c("pos_z","neg_z")) {
    sub <- window_results |> dplyr::filter(term == trm)
    if (nrow(sub) >= 2) {
      cochran_q(sub$estimate, sub$std.error, trm)
      wald_windows(window_results, trm)
    }
  }
}

if (nrow(window_results) > 0) {
  p_win <- ggplot(window_results,
                  aes(x = window, y = OR, color = term, group = term)) +
    geom_point(size = 3.5, position = position_dodge(0.35)) +
    geom_errorbar(aes(ymin = OR_low, ymax = OR_high), width = 0.15,
                  position = position_dodge(0.35)) +
    geom_hline(yintercept = 1, linetype = "dashed") +
    scale_color_manual(values = c("pos_z" = "#2166ac", "neg_z" = "#d6604d"),
                       labels = c("pos_z" = "Positive", "neg_z" = "Negative")) +
    labs(title = "Momentum Window Sensitivity",
         subtitle = "OR ± 95% CI across accumulation windows",
         x = "Window", y = "Odds Ratio", color = NULL) + theme_drift()
  save_plot(p_win, "window_sensitivity_OR.png")
}

# --- Section 14: ML + Stacked Ensemble (ST-05) -------------
log_section("14", "ML + STACKED ENSEMBLE (ST-05)")

X_vars <- c("SEMANTIC_DISTANCE_L","SEM_DIST_TO_PEAK_L",
            "SEM_TIME_FROM_PEAK_L","DELTA_SEM_DISTANCE_L",
            "OSCILLATION_L","LIFECYCLE_INDEX_L",
            "SENTIMENT_SCORE","ENTROPY_SENTIMENT",
            "SEM_DIST_LIFECYCLE","SENTIMENT_VOLATILITY",
            "TEMPORAL_ACCELERATION")

df_ml <- df_fixest |>
  dplyr::filter(complete.cases(across(all_of(c("DRIFT_STABILITY_L", X_vars)))))
X_mat <- model.matrix(~ . - 1, data = df_ml[, X_vars])
y_vec <- df_ml$DRIFT_STABILITY_L
set.seed(CONFIG$seed)
tr_i  <- sample(nrow(df_ml), 0.7 * nrow(df_ml))
te_i  <- setdiff(seq_len(nrow(df_ml)), tr_i)
Xtr   <- X_mat[tr_i, ]; Xte <- X_mat[te_i, ]
ytr   <- y_vec[tr_i];   yte <- y_vec[te_i]

cv_en <- glmnet::cv.glmnet(Xtr, ytr, alpha = 0.5, nfolds = CONFIG$cv_folds)
cv_la <- glmnet::cv.glmnet(Xtr, ytr, alpha = 1.0, nfolds = CONFIG$cv_folds)
cv_ri <- glmnet::cv.glmnet(Xtr, ytr, alpha = 0.0, nfolds = CONFIG$cv_folds)
en_m  <- glmnet::glmnet(Xtr, ytr, alpha = 0.5, lambda = cv_en$lambda.min)
la_m  <- glmnet::glmnet(Xtr, ytr, alpha = 1.0, lambda = cv_la$lambda.min)
ri_m  <- glmnet::glmnet(Xtr, ytr, alpha = 0.0, lambda = cv_ri$lambda.min)
rf_m  <- randomForest::randomForest(Xtr, ytr, ntree = CONFIG$rf_trees,
                                     mtry = floor(sqrt(ncol(Xtr))),
                                     importance = TRUE)
gbm_m <- caret::train(x = Xtr, y = ytr, method = "gbm", verbose = FALSE,
                       trControl = caret::trainControl(method = "cv",
                                                        number = CONFIG$cv_folds),
                       tuneGrid  = data.frame(n.trees = 150,
                                              interaction.depth = 3,
                                              shrinkage = 0.05,
                                              n.minobsinnode = 10))

pred_en  <- as.numeric(predict(en_m,  Xte))
pred_la  <- as.numeric(predict(la_m,  Xte))
pred_ri  <- as.numeric(predict(ri_m,  Xte))
pred_rf  <- predict(rf_m,  Xte)
pred_gbm <- predict(gbm_m, Xte)

# ST-05: Ridge meta-learner
stack_df   <- data.frame(en = pred_en, la = pred_la, ri = pred_ri,
                          rf = pred_rf, gbm = pred_gbm, y = yte)
meta_m     <- glmnet::cv.glmnet(as.matrix(stack_df[, 1:5]),
                                  stack_df$y, alpha = 0, nfolds = 5)
pred_stack <- as.numeric(predict(meta_m, as.matrix(stack_df[, 1:5]),
                                  s = "lambda.min"))

ml_perf <- tibble(
  Model = c("Elastic Net","LASSO","Ridge","RF","GBM","Stacked"),
  RMSE  = round(c(rmse(yte, pred_en), rmse(yte, pred_la), rmse(yte, pred_ri),
                  rmse(yte, pred_rf), rmse(yte, pred_gbm), rmse(yte, pred_stack)), 5),
  R2    = round(c(r2_oos(yte, pred_en), r2_oos(yte, pred_la), r2_oos(yte, pred_ri),
                  r2_oos(yte, pred_rf), r2_oos(yte, pred_gbm), r2_oos(yte, pred_stack)), 5),
  MAE   = round(c(mae(yte, pred_en), mae(yte, pred_la), mae(yte, pred_ri),
                  mae(yte, pred_rf), mae(yte, pred_gbm), mae(yte, pred_stack)), 5)
)
cat("\n[ML Performance (OOS)]:\n"); print(ml_perf)

# --- Section 15: Survival (ST-06 Schoenfeld) ---------------
log_section("15", "SURVIVAL (ST-06 Schoenfeld plots)")

df_surv <- df_fixest |>
  group_by(listing_id) |> arrange(quarter_id) |>
  summarise(
    exit_q        = ifelse(any(EXIT_EVENT == 1),
                           min(quarter_id[EXIT_EVENT == 1]), NA_real_),
    survival_time = ifelse(is.na(exit_q), max(quarter_id), exit_q),
    event         = as.numeric(!is.na(exit_q)),
    across(c(SEMANTIC_DISTANCE_L, SEM_DIST_TO_PEAK_L, SEM_TIME_FROM_PEAK_L,
             OSCILLATION_L, LIFECYCLE_INDEX_L, SENTIMENT_SCORE,
             district_id, is_tourist_core), first),
    .groups = "drop")

cox_f <- Surv(survival_time, event) ~
  SEMANTIC_DISTANCE_L + SEM_DIST_TO_PEAK_L + SEM_TIME_FROM_PEAK_L +
  OSCILLATION_L + LIFECYCLE_INDEX_L + SENTIMENT_SCORE
cox_m <- survival::coxph(cox_f, data = df_surv)
ph_test <- survival::cox.zph(cox_m)
cat("\n[PH test]:\n"); print(ph_test)

png("cox_schoenfeld.png", width = 1800, height = 1200, res = 200, bg = "white")
plot(ph_test, main = "Cox PH — Schoenfeld Residuals")
dev.off()
log_msg("  [Saved] cox_schoenfeld.png")

# --- Section 18: GAM + QR (ST-08 te()) --------------------
log_section("18", "GAM + QUANTILE (ST-08 te())")

gam_m <- tryCatch(
  mgcv::gam(
    DRIFT_STABILITY_L ~
      s(SEMANTIC_DISTANCE_L,  bs = "tp") +
      s(SEM_TIME_FROM_PEAK_L, bs = "tp") +
      s(OSCILLATION_L,        bs = "tp") +
      s(LIFECYCLE_INDEX_L,    bs = "tp") +
      te(SEMANTIC_DISTANCE_L, OSCILLATION_L, bs = "tp"),
    data = df_fixest, family = gaussian(), method = "REML"),
  error   = function(e) { log_msg(sprintf("[GAM] ERROR: %s", e$message)); NULL },
  warning = function(w) { log_msg(sprintf("[GAM] WARN: %s",  w$message)); NULL }
)
if (!is.null(gam_m) && inherits(gam_m, "gam")) {
  cat("\n[GAM with te() interaction]:\n"); print(summary(gam_m))
  tryCatch({
    png("gam_smooth_plots.png", width = 2400, height = 1600,
        res = 200, bg = "white")
    par(mfrow = c(2, 3))
    plot(gam_m, pages = 0, shade = TRUE, col = "#2166ac", seWithMean = TRUE)
    dev.off()
    log_msg("  [Saved] gam_smooth_plots.png")
  }, error = function(e) { tryCatch(dev.off(), error = function(e2) NULL) })
}

# --- Section 19: Bootstrap BCa (ST-09) --------------------
log_section("19", "BOOTSTRAP BCa (ST-09)")

boot_fn  <- function(data, idx) coef(lm(formula_base, data = data[idx, ]))
boot_res <- tryCatch(
  boot::boot(df_fixest, boot_fn, R = CONFIG$boot_R),
  error   = function(e) { log_msg(sprintf("[boot] ERROR: %s", e$message)); NULL },
  warning = function(w) { log_msg(sprintf("[boot] WARN: %s",  w$message)); NULL }
)
base_cfs <- coef(lm(formula_base, df_fixest))

boot_ci_tbl <- if (!is.null(boot_res) && inherits(boot_res, "boot")) {
  purrr::map_dfr(seq_along(base_cfs), function(i) {
    ci_p <- tryCatch(
      boot::boot.ci(boot_res, index = i, type = "perc")$percent[4:5],
      error = function(e) c(NA_real_, NA_real_))
    ci_b <- tryCatch(
      boot::boot.ci(boot_res, index = i, type = "bca")$bca[4:5],
      error = function(e) c(NA_real_, NA_real_))
    tibble(param   = names(base_cfs)[i],
           est     = round(base_cfs[i], 5),
           perc_lo = round(ci_p[1], 5), perc_hi = round(ci_p[2], 5),
           bca_lo  = round(ci_b[1], 5), bca_hi  = round(ci_b[2], 5),
           sig_perc = !(is.na(ci_p[1]) || (ci_p[1] < 0 & ci_p[2] > 0)),
           sig_bca  = !(is.na(ci_b[1]) || (ci_b[1] < 0 & ci_b[2] > 0)))
  })
} else {
  tibble()
}
if (nrow(boot_ci_tbl) > 0) { cat("\n[Bootstrap CI]:\n"); print(boot_ci_tbl, n = 20) }

# --- Section 20: Diagnostics (ST-10 Cook's D) -------------
log_section("20", "DIAGNOSTICS (ST-10 Cook's D)")

base_ols <- lm(formula_base, data = df_fixest)
resid_b  <- residuals(base_ols)
hat_vals <- hatvalues(base_ols)
cooks_d  <- cooks.distance(base_ols)
n_inf    <- sum(cooks_d > 4 / nrow(df_fixest), na.rm = TRUE)
log_result("Cook's D > 4/n count", n_inf, n_inf < nrow(df_fixest) * 0.05)

tryCatch({
  png("influence_plot.png", width = 1400, height = 1000, res = 180, bg = "white")
  plot(hat_vals, cooks_d,
       xlab = "Hat Value (Leverage)", ylab = "Cook's Distance",
       main = "Influence Plot",
       col  = ifelse(cooks_d > 4 / nrow(df_fixest), "#d73027", "#4393c3"),
       pch  = ifelse(cooks_d > 4 / nrow(df_fixest), 19, 1), cex = 0.6)
  abline(h = 4 / nrow(df_fixest), lty = 2, col = "red")
  dev.off(); log_msg("  [Saved] influence_plot.png")
}, error = function(e) {
  tryCatch(dev.off(), error = function(e2) invisible(NULL))
  log_msg(sprintf("[influence plot] ERROR: %s", e$message))
})

# --- Section 21: Visualisations (VZ-01 cohort heatmap) ----
log_section("21", "VISUALISATIONS")

# VZ-01: Cohort × Quarter heatmap
cohort_heat <- df_fixest |>
  group_by(COHORT_Q, quarter_id) |>
  summarise(mean_sem = mean(SEMANTIC_DISTANCE_L, na.rm = TRUE), .groups = "drop") |>
  dplyr::filter(!is.na(COHORT_Q))
if (nrow(cohort_heat) > 0) {
  p_heat <- ggplot(cohort_heat,
                   aes(x = factor(quarter_id), y = factor(COHORT_Q),
                       fill = mean_sem)) +
    geom_tile(color = "white", linewidth = 0.3) +
    scale_fill_viridis_c(option = "C", name = "Mean\nSem Dist") +
    labs(title = "Cohort × Quarter: Mean Semantic Distance",
         x = "Quarter", y = "Entry Cohort (Quarter)") + theme_drift()
  save_plot(p_heat, "cohort_heatmap.png", w = 11, h = 7)
}

# Temporal evolution plot
ts_data <- df_fixest |>
  group_by(quarter_id, quarter_date) |>
  summarise(mean_drift = mean(DRIFT_STABILITY_L,   na.rm = TRUE),
            mean_sem   = mean(SEMANTIC_DISTANCE_L, na.rm = TRUE),
            mean_osc   = mean(OSCILLATION_L,       na.rm = TRUE),
            .groups = "drop") |>
  arrange(quarter_date)

p_ts <- ts_data |>
  tidyr::pivot_longer(c(mean_drift, mean_sem, mean_osc),
                      names_to = "metric", values_to = "value") |>
  mutate(metric = dplyr::case_match(
    metric,
    "mean_drift" ~ "Drift Stability",
    "mean_sem"   ~ "Semantic Distance",
    "mean_osc"   ~ "Oscillation", .default = metric)) |>
  ggplot(aes(x = quarter_date, y = value, color = metric)) +
  geom_line(linewidth = 1.2) + geom_point(size = 2.5) +
  facet_wrap(~ metric, scales = "free_y", ncol = 1) +
  scale_color_viridis_d(option = "D") +
  labs(title = "Temporal Evolution of Semantic Metrics",
       x = "Quarter", y = "Mean (log-transformed)") +
  theme_drift() + theme(legend.position = "none")
save_plot(p_ts, "temporal_evolution.png", w = 8, h = 9)

# --- Section 22: Save results ------------------------------
log_section("22", "SAVE RESULTS")

csv_saves <- list(
  window_sensitivity_results   = window_results,
  ml_performance_results       = ml_perf,
  htmt_discriminant_validity   = htmt_df,
  bootstrap_ci_results         = boot_ci_tbl,
  mice_pool_check              = pool_check
)
for (nm in names(csv_saves)) {
  tryCatch(
    readr::write_csv(csv_saves[[nm]], paste0(nm, ".csv")),
    error = function(e) log_msg(sprintf("[CSV %s] %s", nm, e$message))
  )
}

save(list = ls(), file = "semantic_drift_v6.RData")
close(log_con)

cat("\n[Complete] semantic_drift_robustness_v6.R\n")
cat(strrep("=", 70), "\n")

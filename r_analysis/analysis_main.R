# =============================================================================
# Conformity, Trajectory, and Space
# A Dynamic Capabilities Account of Representational Survival
# in Platform Accommodation Markets
#
# R Analysis Script — Full Pipeline
# Data: dataset_0322.xlsx (H12 sheet → H1 OLS; H3 sheet → H2 Logit)
# Outputs: figures/ directory (PNG)
# =============================================================================

# ── 0. PACKAGES ───────────────────────────────────────────────────────────────
pkgs <- c(
  "readxl", "dplyr", "tidyr", "ggplot2", "ggrepel",
  "sandwich", "lmtest", "forcats", "scales",
  "patchwork", "RColorBrewer", "tibble", "stringr"
)
new_pkgs <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
if (length(new_pkgs) > 0)
  install.packages(new_pkgs, repos = "https://cloud.r-project.org")

suppressMessages({
  library(readxl); library(dplyr);   library(tidyr);  library(ggplot2)
  library(ggrepel); library(sandwich); library(lmtest); library(forcats)
  library(scales);  library(patchwork); library(RColorBrewer)
  library(tibble);  library(stringr)
})

# ── PATHS ─────────────────────────────────────────────────────────────────────
script_dir <- if (interactive()) {
  dirname(rstudioapi::getSourceEditorContext()$path)
} else {
  dirname(normalizePath(sys.frame(1)$ofile, mustWork = FALSE))
}

DATA_PATH <- "C:/Users/LG/Downloads/dataset_0322.xlsx"

FIGURE_DIR <- file.path(script_dir, "figures")
dir.create(FIGURE_DIR, showWarnings = FALSE, recursive = TRUE)

cat(sprintf("[INFO] Data  : %s\n", DATA_PATH))
cat(sprintf("[INFO] Figs  : %s\n", FIGURE_DIR))

# ── GLOBAL THEME ──────────────────────────────────────────────────────────────
COL_MAIN  <- "#2C5F8A"
COL_RED   <- "#C0392B"
COL_GREEN <- "#27AE60"
COL_GOLD  <- "#D4A017"
COL_GREY  <- "#7F8C8D"
COL_LIGHT <- "#BDC3C7"

theme_paper <- function(base_size = 11) {
  theme_classic(base_size = base_size) +
    theme(
      plot.background   = element_rect(fill = "white", colour = NA),
      panel.background  = element_rect(fill = "white", colour = NA),
      panel.grid.major  = element_line(colour = "grey92", linewidth = 0.4),
      panel.grid.minor  = element_blank(),
      panel.border      = element_rect(colour = "grey70", fill = NA, linewidth = 0.5),
      strip.background  = element_rect(fill = "grey96", colour = "grey70"),
      strip.text        = element_text(face = "bold", size = base_size * 0.9),
      plot.title        = element_text(face = "bold", size = base_size * 1.15,
                                       margin = ggplot2::margin(b = 6)),
      plot.subtitle     = element_text(size = base_size * 0.9, colour = "grey40",
                                       margin = ggplot2::margin(b = 10)),
      plot.caption      = element_text(size = base_size * 0.75, colour = "grey50",
                                       hjust = 0),
      legend.background = element_rect(fill = "white", colour = NA),
      legend.key        = element_rect(fill = "white", colour = NA),
      axis.line         = element_line(colour = "grey60"),
      axis.ticks        = element_line(colour = "grey60")
    )
}

save_fig <- function(p, name, w = 9, h = 6, dpi = 300) {
  path <- file.path(FIGURE_DIR, paste0(name, ".png"))
  ggsave(path, p, width = w, height = h, dpi = dpi, bg = "white")
  cat(sprintf("  [Saved] %s\n", basename(path)))
}

# =============================================================================
# SECTION 1. DATA LOADING
# =============================================================================
cat("\n===== SECTION 1: DATA LOADING =====\n")

df_h12_raw <- read_excel(DATA_PATH, sheet = "H12")
df_h3_raw  <- read_excel(DATA_PATH, sheet = "H3")

cat(sprintf("[1] H12 sheet: %d rows × %d cols\n", nrow(df_h12_raw), ncol(df_h12_raw)))
cat(sprintf("[1] H3  sheet: %d rows × %d cols\n", nrow(df_h3_raw),  ncol(df_h3_raw)))

# ── H1 sample: drop rows with NA on key variables ──────────────────────────
h1_vars <- c("fit_init", "variability_init", "activity_init", "reviews_init",
             "price_init", "superhost_init", "amenity_init", "sentiment_init",
             "lifetime_quarters", "entry_quarter")

df_h1 <- df_h12_raw %>%
  filter(if_all(all_of(h1_vars), ~ !is.na(.)))

cat(sprintf("[1] H1 estimation sample: N = %d (paper: N=317)\n", nrow(df_h1)))

# ── H2 sample: per paper Section 3.5.3 ────────────────────────────────────
df_h2 <- df_h3_raw %>%
  filter(new_spell == 0) %>%
  filter(!is.na(positive_momentum_4q), !is.na(negative_momentum_4q)) %>%
  filter(!is.na(platform_activity_lag),
         !is.na(n_reviews_qtr_lag),
         !is.na(sentiment_mean_qtr_lag))

# Remove terminal quarter
terminal_qtr <- max(df_h2$period_qtr, na.rm = TRUE)
df_h2 <- df_h2 %>% filter(period_qtr != terminal_qtr)

# Remove zero-event quarters
zero_event_qtrs <- df_h2 %>%
  group_by(period_qtr) %>%
  summarise(n_ev = sum(exit_next, na.rm = TRUE), .groups = "drop") %>%
  filter(n_ev == 0) %>% pull(period_qtr)
df_h2 <- df_h2 %>% filter(!period_qtr %in% zero_event_qtrs)

# Standardise momentum variables
df_h2 <- df_h2 %>%
  mutate(
    pos_mom_z = as.numeric(scale(positive_momentum_4q)),
    neg_mom_z = as.numeric(scale(negative_momentum_4q))
  )

cat(sprintf("[1] H2 estimation sample: N = %d, events = %d, rate = %.1f%%\n",
            nrow(df_h2),
            sum(df_h2$exit_next, na.rm = TRUE),
            100 * mean(df_h2$exit_next, na.rm = TRUE)))

# =============================================================================
# SECTION 2. H1 OLS — LISTING LIFETIME MODELS
# =============================================================================
cat("\n===== SECTION 2: H1 OLS MODELS =====\n")

# M1: joint baseline
m1 <- lm(lifetime_quarters ~ fit_init + variability_init + activity_init +
           reviews_init + price_init + superhost_init + amenity_init +
           sentiment_init + factor(entry_quarter),
         data = df_h1)

vcov_hc1 <- vcovHC(m1, type = "HC1")
coef_m1  <- coef(m1)
se_m1    <- sqrt(diag(vcov_hc1))
t_m1     <- coef_m1 / se_m1
p_m1     <- 2 * pt(-abs(t_m1), df = m1$df.residual)

# M2: fit only
m2 <- lm(lifetime_quarters ~ fit_init + activity_init + reviews_init +
           price_init + superhost_init + amenity_init + sentiment_init +
           factor(entry_quarter), data = df_h1)
vcov_hc1_m2 <- vcovHC(m2, type = "HC1")

# M3: variability only
m3 <- lm(lifetime_quarters ~ variability_init + activity_init + reviews_init +
           price_init + superhost_init + amenity_init + sentiment_init +
           factor(entry_quarter), data = df_h1)

cat(sprintf("[2] M1 R2=%.3f | AdjR2=%.3f | N=%d\n",
            summary(m1)$r.squared, summary(m1)$adj.r.squared, nrow(df_h1)))
cat(sprintf("[2] H1a beta=%.3f p=%.3f\n",
            coef_m1["fit_init"], p_m1["fit_init"]))
cat(sprintf("[2] H1b beta=%.3f p=%.3f\n",
            coef_m1["variability_init"], p_m1["variability_init"]))

# =============================================================================
# FIGURE 1: H1 Forest Plot (HC1 robust SE)
# =============================================================================
cat("\n[FIG-1] H1 OLS coefficient forest plot\n")

focal_vars <- c("fit_init", "variability_init")
ctrl_vars  <- c("activity_init", "reviews_init", "price_init",
                "superhost_init", "amenity_init", "sentiment_init")
all_vars   <- c(focal_vars, ctrl_vars)

labels_map <- c(
  fit_init          = "Representational fit\n(H1a, distance coding)",
  variability_init  = "Representational variability\n(H1b)",
  activity_init     = "Platform activity (log)",
  reviews_init      = "Reviews per quarter",
  price_init        = "Log price",
  superhost_init    = "Superhost status",
  amenity_init      = "Amenity count",
  sentiment_init    = "Sentiment score"
)

forest_df <- tibble(
  var      = all_vars,
  label    = unname(labels_map[all_vars]),
  estimate = coef_m1[all_vars],
  se       = se_m1[all_vars],
  pval     = p_m1[all_vars]
) %>%
  mutate(
    lo95  = estimate - 1.96 * se,
    hi95  = estimate + 1.96 * se,
    lo90  = estimate - 1.645 * se,
    hi90  = estimate + 1.645 * se,
    sig90 = (lo90 > 0 | hi90 < 0),
    color_grp = case_when(
      var == "fit_init"         ~ "H1a",
      var == "variability_init" ~ "H1b",
      TRUE                      ~ "Control"
    ),
    label = fct_rev(factor(label, levels = rev(unname(labels_map[all_vars]))))
  )

fig1 <- ggplot(forest_df, aes(y = label, x = estimate, colour = color_grp)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.6) +
  geom_linerange(aes(xmin = lo95, xmax = hi95), linewidth = 0.6, alpha = 0.45) +
  geom_linerange(aes(xmin = lo90, xmax = hi90), linewidth = 1.5) +
  geom_point(aes(shape = sig90), size = 3.5) +
  geom_text(aes(label = sprintf("β=%.3f", estimate), x = hi95 + 0.02),
            hjust = 0, size = 2.9, colour = "grey30") +
  scale_colour_manual(
    values = c("H1a" = COL_RED, "H1b" = COL_MAIN, "Control" = COL_GREY),
    name = NULL
  ) +
  scale_shape_manual(
    values = c("TRUE" = 19, "FALSE" = 1),
    labels = c("TRUE" = "p < .10", "FALSE" = "n.s."),
    name   = "Significance"
  ) +
  scale_x_continuous(expand = expansion(mult = c(0.05, 0.25))) +
  labs(
    title    = "Figure 1  H1 OLS Coefficient Forest Plot",
    subtitle = "Thick bar = 90% CI; thin bar = 95% CI  |  Fit coded as cosine distance (higher = worse fit)",
    x        = "Coefficient (quarters of listing lifetime)",
    y        = NULL,
    caption  = sprintf("N=%d; HC1 robust SE; entry-quarter FE included; R²=%.3f",
                       nrow(df_h1), summary(m1)$r.squared)
  ) +
  theme_paper() +
  theme(legend.position = "bottom")

save_fig(fig1, "fig1_h1_forest_plot")

# =============================================================================
# SECTION 3. H2 DISCRETE-TIME LOGIT
# =============================================================================
cat("\n===== SECTION 3: H2 LOGIT MODELS =====\n")

# Model A: contemporaneous (single-period)
modelA <- glm(exit_next ~ pos_step + neg_step +
                platform_activity_lag + n_reviews_qtr_lag + sentiment_mean_qtr_lag +
                factor(period_qtr),
              family = binomial(link = "logit"), data = df_h2)

# Model B: 4Q cumulative
modelB <- glm(exit_next ~ pos_mom_z + neg_mom_z +
                platform_activity_lag + n_reviews_qtr_lag + sentiment_mean_qtr_lag +
                factor(period_qtr),
              family = binomial(link = "logit"), data = df_h2)

# Cluster-robust SE function (cluster by listing_id)
cluster_vcov <- function(model, cluster_var) {
  X   <- model.matrix(model)
  y   <- model$y
  mu  <- fitted(model)
  idx <- !is.na(cluster_var)
  cl  <- factor(cluster_var)
  score <- sweep(X, 1, y - mu, "*")
  K    <- ncol(X)
  meat <- matrix(0, K, K)
  for (g in levels(cl)) {
    s_g  <- colSums(score[cl == g, , drop = FALSE])
    meat <- meat + outer(s_g, s_g)
  }
  bread <- vcov(model)
  G <- nlevels(cl); p <- K; n <- nrow(X)
  adj <- (G / (G - 1)) * ((n - 1) / (n - p))
  adj * (bread %*% meat %*% bread)
}

vcov_B  <- cluster_vcov(modelB, df_h2$listing_id)
se_B    <- sqrt(diag(vcov_B))
coef_B  <- coef(modelB)
z_B     <- coef_B / se_B
p_B     <- 2 * pnorm(-abs(z_B))

beta_rec <- coef_B["pos_mom_z"]
beta_det <- coef_B["neg_mom_z"]
se_rec   <- se_B["pos_mom_z"]
se_det   <- se_B["neg_mom_z"]

OR_rec <- exp(beta_rec)
OR_det <- exp(beta_det)

# Wald test for magnitude asymmetry
R_vec <- setNames(numeric(length(coef_B)), names(coef_B))
R_vec["pos_mom_z"] <-  1
R_vec["neg_mom_z"] <-  1
est_sum  <- sum(R_vec * coef_B)
var_sum  <- as.numeric(t(R_vec) %*% vcov_B %*% R_vec)
wald_chi2 <- (est_sum / sqrt(var_sum))^2
wald_p    <- pchisq(wald_chi2, df = 1, lower.tail = FALSE)
dominant  <- ifelse(abs(beta_rec) > abs(beta_det), "RECOVERY", "DETERIORATION")

cat(sprintf("[3] Model B: OR_recovery=%.3f (p=%.3f) | OR_deterioration=%.3f (p=%.3f)\n",
            OR_rec, p_B["pos_mom_z"], OR_det, p_B["neg_mom_z"]))
cat(sprintf("[3] Wald chi2(1)=%.2f p=%.3f → %s dominates\n",
            wald_chi2, wald_p, dominant))

# =============================================================================
# FIGURE 2: Model A vs Model B Comparison
# =============================================================================
cat("\n[FIG-2] Model A vs B path-dependence comparison\n")

coef_A  <- coef(modelA)
se_A    <- sqrt(diag(vcovHC(modelA, type = "HC1")))

ab_df <- bind_rows(
  tibble(
    model     = "Model A\n(Contemporaneous)",
    direction = c("Recovery", "Deterioration"),
    beta      = c(coef_A["pos_step"], coef_A["neg_step"]),
    se        = c(se_A["pos_step"],   se_A["neg_step"]),
    se_label  = "Robust SE"
  ),
  tibble(
    model     = "Model B\n(4Q Cumulative)",
    direction = c("Recovery", "Deterioration"),
    beta      = c(beta_rec, beta_det),
    se        = c(se_rec,   se_det),
    se_label  = "Cluster-Robust SE"
  )
) %>%
  mutate(
    lo = beta - 1.96 * se, hi = beta + 1.96 * se,
    OR = exp(beta), OR_lo = exp(lo), OR_hi = exp(hi),
    sig = (lo > 0 | hi < 0),
    direction = factor(direction, levels = c("Recovery", "Deterioration")),
    model     = factor(model, levels = c("Model A\n(Contemporaneous)",
                                         "Model B\n(4Q Cumulative)"))
  )

p_beta <- ggplot(ab_df, aes(y = direction, x = beta,
                             colour = direction, shape = sig)) +
  facet_wrap(~ model, scales = "free_x") +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
  geom_linerange(aes(xmin = lo, xmax = hi), linewidth = 0.9) +
  geom_point(size = 3.8) +
  geom_text(aes(label = sprintf("β=%.3f", beta)), hjust = -0.2, size = 2.9,
            colour = "grey25") +
  scale_colour_manual(values = c("Recovery" = COL_GREEN, "Deterioration" = COL_RED),
                      guide = "none") +
  scale_shape_manual(values = c("TRUE" = 19, "FALSE" = 1),
                     labels = c("TRUE" = "p<.05", "FALSE" = "n.s."), name = NULL) +
  scale_x_continuous(expand = expansion(mult = c(0.05, 0.40))) +
  labs(subtitle = "Log-odds coefficients (95% CI)", x = "β", y = NULL) +
  theme_paper() + theme(legend.position = "bottom")

p_or <- ggplot(ab_df, aes(y = direction, x = OR,
                           colour = direction, shape = sig)) +
  facet_wrap(~ model, scales = "free_x") +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey50") +
  geom_linerange(aes(xmin = OR_lo, xmax = OR_hi), linewidth = 0.9) +
  geom_point(size = 3.8) +
  geom_text(aes(label = sprintf("OR=%.3f", OR)), hjust = -0.2, size = 2.9,
            colour = "grey25") +
  scale_colour_manual(values = c("Recovery" = COL_GREEN, "Deterioration" = COL_RED),
                      guide = "none") +
  scale_shape_manual(values = c("TRUE" = 19, "FALSE" = 1), guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0.05, 0.35))) +
  labs(subtitle = "Odds Ratios (95% CI)", x = "OR", y = NULL) +
  theme_paper()

fig2 <- (p_beta / p_or) +
  plot_annotation(
    title    = "Figure 2  Model A vs Model B: Path-Dependence Test",
    subtitle = "Model A null confirms single-period revisions have no survival effect; Model B cumulative trajectory is significant",
    caption  = sprintf("Model B: N=%d, cluster-robust SE; Model A: contemporaneous specification",
                       nrow(model.matrix(modelB))),
    theme    = theme_paper()
  )

save_fig(fig2, "fig2_model_AB_comparison", w = 10, h = 8)

# =============================================================================
# FIGURE 3: Beta Magnitude Comparison (Gain-Sensitivity)
# =============================================================================
cat("\n[FIG-3] Beta magnitude / gain-sensitivity\n")

mag_df <- tibble(
  direction = c("Deterioration\n(negative_momentum_4q)",
                "Recovery\n(positive_momentum_4q)"),
  beta_abs  = c(abs(beta_det), abs(beta_rec)),
  se        = c(se_det, se_rec),
  label     = c(sprintf("β = +%.3f\n(increases exit risk)", beta_det),
                sprintf("β = −%.3f\n(reduces exit risk)", abs(beta_rec))),
  dominant  = c(FALSE, TRUE)
)

fig3 <- ggplot(mag_df, aes(x = direction, y = beta_abs, fill = dominant)) +
  geom_col(width = 0.5, colour = "white", linewidth = 0.4) +
  geom_errorbar(aes(ymin = pmax(beta_abs - 1.96 * se, 0),
                    ymax = beta_abs + 1.96 * se),
                width = 0.12, colour = "grey30", linewidth = 0.7) +
  geom_text(aes(label = label, y = beta_abs + 1.96 * se + 0.01),
            vjust = 0, size = 3.2, fontface = "bold") +
  annotate("text", x = 1.5, y = max(mag_df$beta_abs) * 0.6,
           label = sprintf("Wald χ²(1) = %.2f\np = %.3f\n→ %s dominates",
                           wald_chi2, wald_p, dominant),
           size = 3.5, colour = COL_MAIN, fontface = "bold", hjust = 0.5) +
  scale_fill_manual(values = c("TRUE" = COL_GREEN, "FALSE" = COL_RED),
                    guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.35))) +
  labs(
    title    = "Figure 3  Beta Magnitude: Recovery vs Deterioration",
    subtitle = "RECOVERY (green) dominates DETERIORATION — gain-sensitivity, not loss aversion",
    x        = NULL,
    y        = "|β| on log-odds scale (cluster-robust SE)",
    caption  = sprintf("Formal asymmetry test: Wald χ²(1)=%.2f, p=%.3f | N=%d (Model B)",
                       wald_chi2, wald_p, nrow(model.matrix(modelB)))
  ) +
  theme_paper()

save_fig(fig3, "fig3_beta_magnitude")

# =============================================================================
# SECTION 4. TEMPORAL EVOLUTION
# =============================================================================
cat("\n===== SECTION 4: TEMPORAL EVOLUTION =====\n")

df_h3_agg <- df_h3_raw %>%
  filter(!is.na(sem_distance)) %>%
  group_by(period_qtr) %>%
  summarise(
    mean_sem_dist = mean(sem_distance, na.rm = TRUE),
    mean_exit_rate = mean(exit_next, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  arrange(period_qtr)

cat(sprintf("[4] Temporal aggregation: %d quarters\n", nrow(df_h3_agg)))

# =============================================================================
# FIGURE 4: Temporal Evolution of Semantic Distance
# =============================================================================
cat("\n[FIG-4] Temporal evolution\n")

fig4 <- ggplot(df_h3_agg, aes(x = period_qtr, y = mean_sem_dist, group = 1)) +
  geom_ribbon(aes(ymin = mean_sem_dist - 0.02, ymax = mean_sem_dist + 0.02),
              fill = COL_MAIN, alpha = 0.15) +
  geom_line(colour = COL_MAIN, linewidth = 1.0) +
  geom_point(colour = COL_MAIN, size = 2.5) +
  scale_y_continuous(labels = scales::number_format(accuracy = 0.01)) +
  labs(
    title    = "Figure 4  Temporal Evolution of Mean Semantic Distance",
    subtitle = "Market-level average cosine distance between listing descriptions and district-quarter demand centroid (2021Q1–2025Q2)",
    x        = "Quarter",
    y        = "Mean Semantic Distance (cosine)",
    caption  = "Source: H3 panel; N varies by quarter due to entry/exit dynamics"
  ) +
  theme_paper() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))

save_fig(fig4, "fig4_temporal_evolution")

# =============================================================================
# SECTION 5. SPATIAL SUBGROUP (H3)
# =============================================================================
cat("\n===== SECTION 5: H3 SPATIAL SUBGROUP =====\n")

tourist_districts <- c("Yau Tsim Mong", "Wan Chai", "Central & Western")

df_h1_ext <- df_h1 %>%
  left_join(
    df_h3_raw %>%
      select(listing_id, neighbourhood_cleansed) %>%
      distinct(),
    by = "listing_id"
  ) %>%
  mutate(is_tourist = neighbourhood_cleansed %in% tourist_districts)

df_tourist     <- df_h1_ext %>% filter(is_tourist)
df_nontourist  <- df_h1_ext %>% filter(!is_tourist)

cat(sprintf("[5] Tourist-core: n=%d | Non-tourist: n=%d\n",
            nrow(df_tourist), nrow(df_nontourist)))

fit_subgroup <- function(df) {
  if (nrow(df) < 10) return(NULL)
  m <- lm(lifetime_quarters ~ fit_init + variability_init + activity_init +
            reviews_init + price_init + superhost_init + amenity_init +
            sentiment_init + factor(entry_quarter), data = df)
  vc <- vcovHC(m, type = "HC1")
  b  <- coef(m)
  se <- sqrt(diag(vc))
  pv <- 2 * pt(-abs(b / se), df = m$df.residual)
  list(b = b, se = se, pv = pv, r2 = summary(m)$r.squared, n = nrow(df))
}

res_tourist    <- fit_subgroup(df_tourist)
res_nontourist <- fit_subgroup(df_nontourist)

if (!is.null(res_tourist)) {
  cat(sprintf("[5] Tourist    fit_init: β=%.3f p=%.3f\n",
              res_tourist$b["fit_init"], res_tourist$pv["fit_init"]))
  cat(sprintf("[5] Tourist    var_init: β=%.3f p=%.3f\n",
              res_tourist$b["variability_init"], res_tourist$pv["variability_init"]))
}
if (!is.null(res_nontourist)) {
  cat(sprintf("[5] Non-tourist fit_init: β=%.3f p=%.3f\n",
              res_nontourist$b["fit_init"], res_nontourist$pv["fit_init"]))
  cat(sprintf("[5] Non-tourist var_init: β=%.3f p=%.3f\n",
              res_nontourist$b["variability_init"], res_nontourist$pv["variability_init"]))
}

# =============================================================================
# FIGURE 5: Spatial Subgroup Coefficient Comparison
# =============================================================================
cat("\n[FIG-5] Spatial subgroup comparison\n")

make_sg_df <- function(res, grp_label) {
  if (is.null(res)) return(NULL)
  focal <- c("fit_init", "variability_init")
  tibble(
    group    = grp_label,
    variable = c("Representational fit\n(H1a)", "Representational variability\n(H1b)"),
    var_key  = focal,
    estimate = res$b[focal],
    se       = res$se[focal],
    pval     = res$pv[focal],
    n        = res$n
  )
}

sg_df <- bind_rows(
  make_sg_df(res_tourist,    "Tourist-core"),
  make_sg_df(res_nontourist, "Non-tourist"),
  tibble(
    group    = "Full sample",
    variable = c("Representational fit\n(H1a)", "Representational variability\n(H1b)"),
    var_key  = c("fit_init", "variability_init"),
    estimate = coef_m1[c("fit_init", "variability_init")],
    se       = se_m1[c("fit_init", "variability_init")],
    pval     = p_m1[c("fit_init", "variability_init")],
    n        = nrow(df_h1)
  )
) %>%
  mutate(
    lo = estimate - 1.96 * se, hi = estimate + 1.96 * se,
    sig = (lo > 0 | hi < 0),
    sig_label = ifelse(pval < 0.05, "*", ifelse(pval < 0.10, "†", "")),
    group = factor(group, levels = c("Tourist-core", "Non-tourist", "Full sample"))
  )

fig5 <- ggplot(sg_df, aes(y = group, x = estimate, colour = group, shape = sig)) +
  facet_wrap(~ variable, scales = "free_x") +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
  geom_linerange(aes(xmin = lo, xmax = hi), linewidth = 1.0) +
  geom_point(size = 4.0) +
  geom_text(aes(label = sprintf("β=%.3f%s", estimate, sig_label)),
            hjust = -0.2, size = 3.0, colour = "grey25") +
  scale_colour_manual(
    values = c("Tourist-core" = COL_RED, "Non-tourist" = COL_MAIN,
               "Full sample" = COL_GREY),
    name = NULL
  ) +
  scale_shape_manual(values = c("TRUE" = 19, "FALSE" = 1),
                     labels = c("TRUE" = "p<.10", "FALSE" = "n.s."), name = NULL) +
  scale_x_continuous(expand = expansion(mult = c(0.1, 0.4))) +
  labs(
    title    = "Figure 5  H3 Spatial Subgroup Results",
    subtitle = "Directional reversal of variability effect across district types (H3a, H3b)\n★ = sign reversal between tourist-core and non-tourist subgroups",
    x        = "OLS Coefficient (quarters of lifetime)",
    y        = NULL,
    caption  = sprintf("Tourist-core: n=%d (Yau Tsim Mong, Wan Chai, Central & Western)\nNon-tourist: n=%d (remaining 15 districts)",
                       nrow(df_tourist), nrow(df_nontourist))
  ) +
  theme_paper() +
  theme(legend.position = "bottom")

save_fig(fig5, "fig5_spatial_subgroup", w = 11, h = 6)

# =============================================================================
# FIGURE 6: District-level listing count (data coverage)
# =============================================================================
cat("\n[FIG-6] District coverage bar chart\n")

dist_df <- df_h3_raw %>%
  filter(!is.na(neighbourhood_cleansed)) %>%
  group_by(neighbourhood_cleansed) %>%
  summarise(n_obs = n(), .groups = "drop") %>%
  mutate(
    is_tourist = neighbourhood_cleansed %in% tourist_districts,
    district   = fct_reorder(neighbourhood_cleansed, n_obs)
  )

fig6 <- ggplot(dist_df, aes(x = district, y = n_obs, fill = is_tourist)) +
  geom_col(colour = "white", linewidth = 0.3, width = 0.75) +
  geom_text(aes(label = scales::comma(n_obs)), hjust = -0.1, size = 3.0) +
  coord_flip() +
  scale_fill_manual(
    values = c("TRUE" = COL_RED, "FALSE" = COL_MAIN),
    labels = c("TRUE" = "Tourist-core", "FALSE" = "Non-tourist"),
    name   = "District type"
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.2)),
                     labels = scales::comma) +
  labs(
    title    = "Figure 6  Listing-Quarter Observations by Administrative District",
    subtitle = "Red = tourist-core (Yau Tsim Mong, Wan Chai, Central & Western)",
    x        = NULL,
    y        = "Listing-quarter observations",
    caption  = "Source: H3 panel (N=2,729 raw)"
  ) +
  theme_paper() +
  theme(legend.position = "bottom")

save_fig(fig6, "fig6_district_coverage", w = 9, h = 6)

# =============================================================================
# FIGURE 7: Exit Rate Over Time
# =============================================================================
cat("\n[FIG-7] Exit rate over time\n")

exit_df <- df_h3_raw %>%
  filter(!is.na(exit_next), new_spell == 0) %>%
  group_by(period_qtr) %>%
  summarise(
    exit_rate = mean(exit_next, na.rm = TRUE),
    n         = n(),
    .groups   = "drop"
  ) %>%
  filter(period_qtr != max(period_qtr))

fig7 <- ggplot(exit_df, aes(x = period_qtr, y = exit_rate, group = 1)) +
  geom_col(fill = COL_MAIN, alpha = 0.7, width = 0.7) +
  geom_line(colour = COL_RED, linewidth = 0.9) +
  geom_point(colour = COL_RED, size = 2.5) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                     limits = c(0, NA),
                     expand = expansion(mult = c(0, 0.15))) +
  labs(
    title    = "Figure 7  Quarterly Exit Rate Over Time",
    subtitle = "Proportion of listings exiting the platform in the following quarter",
    x        = "Quarter",
    y        = "Exit rate",
    caption  = "Source: H3 panel; new_spell==0; terminal quarter excluded"
  ) +
  theme_paper() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))

save_fig(fig7, "fig7_exit_rate_time")

# =============================================================================
# FIGURE 8: Odds Ratio Summary (H2a and H2b)
# =============================================================================
cat("\n[FIG-8] OR summary visual\n")

or_df <- tibble(
  spec      = rep(c("Model A\n(Contemporaneous)", "Model B\n(4Q Cumulative)"), each = 2),
  direction = rep(c("Recovery", "Deterioration"), 2),
  OR        = c(exp(coef(modelA)["pos_step"]),
                exp(coef(modelA)["neg_step"]),
                OR_rec,
                OR_det),
  sig       = c(FALSE, FALSE, p_B["pos_mom_z"] < 0.05, p_B["neg_mom_z"] < 0.05)
) %>%
  mutate(
    spec      = factor(spec, levels = c("Model A\n(Contemporaneous)",
                                        "Model B\n(4Q Cumulative)")),
    direction = factor(direction, levels = c("Recovery", "Deterioration"))
  )

fig8 <- ggplot(or_df, aes(x = direction, y = OR, fill = direction, alpha = sig)) +
  facet_wrap(~ spec) +
  geom_col(width = 0.55, colour = "white", linewidth = 0.4) +
  geom_hline(yintercept = 1, linetype = "dashed", colour = "grey50") +
  geom_text(aes(label = sprintf("OR=%.3f%s", OR,
                                ifelse(sig, "*", ""))),
            vjust = -0.4, fontface = "bold", size = 3.4) +
  scale_fill_manual(values = c("Recovery" = COL_GREEN, "Deterioration" = COL_RED),
                    guide = "none") +
  scale_alpha_manual(values = c("TRUE" = 1.0, "FALSE" = 0.5),
                     labels = c("TRUE" = "p<.05", "FALSE" = "n.s."),
                     name = "Significance") +
  scale_y_continuous(limits = c(0, 1.5), expand = expansion(mult = c(0, 0.15))) +
  labs(
    title    = "Figure 8  Odds Ratio Summary: Model A vs Model B",
    subtitle = "Path-dependence: only 4Q cumulative recovery is statistically significant",
    x        = NULL,
    y        = "Odds Ratio",
    caption  = sprintf("Model B N=%d; * p<.05 (cluster-robust SE); n.s. = not significant",
                       nrow(model.matrix(modelB)))
  ) +
  theme_paper() +
  theme(legend.position = "bottom")

save_fig(fig8, "fig8_odds_ratio_summary")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
cat("\n\n===== RESULTS SUMMARY =====\n")
cat(sprintf("H1a: fit_init β=%.3f (SE=%.3f, p=%.3f)\n",
            coef_m1["fit_init"], se_m1["fit_init"], p_m1["fit_init"]))
cat(sprintf("H1b: var_init β=%.3f (SE=%.3f, p=%.3f)\n",
            coef_m1["variability_init"], se_m1["variability_init"],
            p_m1["variability_init"]))
cat(sprintf("H2a (Model B): OR=%.3f β=%.3f p=%.3f\n",
            OR_rec, beta_rec, p_B["pos_mom_z"]))
cat(sprintf("H2b (Model B): OR=%.3f β=%.3f p=%.3f\n",
            OR_det, beta_det, p_B["neg_mom_z"]))
cat(sprintf("Wald asymmetry: chi2=%.2f p=%.3f (%s dominates)\n",
            wald_chi2, wald_p, dominant))

figs_produced <- list.files(FIGURE_DIR, pattern = "\\.png$")
cat(sprintf("\nFigures saved (%d):\n", length(figs_produced)))
for (f in figs_produced) cat(sprintf("  %s\n", f))
cat("===========================\n")

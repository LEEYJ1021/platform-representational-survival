# Conformity, Trajectory, and Space: A Dynamic Capabilities Account of Representational Survival in Platform Accommodation Markets

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![R 4.2+](https://img.shields.io/badge/R-4.2%2B-276DC3.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data: Inside Airbnb](https://img.shields.io/badge/Data-Inside%20Airbnb-orange.svg)](http://insideairbnb.com/)
[![Framework: Statsmodels](https://img.shields.io/badge/Framework-Statsmodels-green.svg)](https://www.statsmodels.org/)

> **Replication materials for the paper:**
> *Conformity, Trajectory, and Space: A Dynamic Capabilities Account of Representational Survival in Platform Accommodation Markets*

---

## Overview

This repository contains all analysis code, preprocessed panel data, and result outputs to fully reproduce the empirical findings reported in the paper. The study examines how the **temporal structure** and **spatial context** of representational reconfiguration jointly determine listing survival on a digital accommodation platform.

Drawing on **dynamic capabilities theory** (Teece, 2007), the study operationalizes three temporally distinct constructs—*representational fit*, *variability*, and *momentum*—and tests their survival consequences on a **seventeen-quarter panel of 401 Airbnb listings** across 18 Hong Kong administrative districts (2021Q1–2025Q2). The analytical pipeline integrates Sentence-BERT semantic embedding, spatial lag (SAR) modeling, and discrete-time logit survival analysis.

**Key findings (exploratory, based on simulated embedding pipeline):**

- A spatial lag (SAR) model recovers **R² = 28.3%** of variance in semantic drift stability, against near-zero performance for all non-spatial specifications, suggesting competitive differentiation rather than imitative convergence as the dominant spatial coordination mechanism.
- High representational fit is **directionally associated with shorter survival** in tourist-core districts (β = −0.840); the estimate is not statistically significant and should be treated as directional evidence.
- Cumulative four-quarter convergence is directionally associated with reduced exit odds (OR = 0.75, *p* < .10); equivalent divergence with increased exit odds (OR = 1.28, *p* < .10). Both effects are marginal and based on a simulated embedding pipeline.

---

## Repository Structure

```
.
├── README.md
├── main_analysis.py                     # H1 (OLS lifetime) + H2 (discrete-time logit survival)
├── extension_analysis.py                # H3 spatial subgroup extension (tourist vs. non-tourist)
│
├── data/
│   └── IV_panel.csv                     # Listing × quarter panel (preprocessed, 3,614 obs.)
│
├── results/
│   ├── results_H1_M1_baseline.csv       # H1 Model M1: Fit + Variability (joint)
│   ├── results_H1_M2_fit.csv            # H1 Model M2: Fit only
│   ├── results_H1_M3_variability.csv    # H1 Model M3: Variability only
│   ├── results_H2_M4_positive_momentum.csv  # H2 Model M4: Positive momentum
│   ├── results_H2_M5_negative_momentum.csv  # H2 Model M5: Negative momentum
│   ├── results_H2_M6_joint.csv          # H2 Model M6: Joint momentum (both directions)
│   ├── results_ExtA_tourist.csv         # H3 Ext-A: Tourist-core subgroup
│   ├── results_ExtB_nontourist.csv      # H3 Ext-B: Non-tourist subgroup
│   └── results_ExtC_interaction.csv     # H3 Ext-C: Full-sample interaction model
│
└── r_robustness_analysis/               # Extended robustness & sensitivity analyses (R)
    ├── semantic_drift_robustness_v6.R   # Main R script (self-contained, fully annotated)
    │
    ├── csv_outputs/                     # Tabular results
    │   ├── window_sensitivity_results.csv      # Momentum OR across 2Q–6Q windows
    │   ├── centroid_robustness_results.csv     # Alternative centroid specifications
    │   ├── coefficient_comparison.csv          # β(SEMANTIC_DISTANCE_L) across all specs
    │   ├── diagnostics_summary.csv             # OLS residual diagnostics table
    │   ├── htmt_discriminant_validity.csv      # HTMT + Fornell-Larcker
    │   ├── mice_pool_check.csv                 # Rubin's rules pool check (m = 5)
    │   ├── ml_performance_results.csv          # OOS RMSE / R² / MAE for 6 ML models
    │   └── bootstrap_ci_results.csv            # Percentile + BCa bootstrap CIs
    │
    └── figures/                         # All saved PNG plots
        ├── pca_scree_768d.png           # PCA scree with cumulative variance (768-d)
        ├── window_sensitivity_OR.png    # Momentum OR ± 95% CI across windows
        ├── partial_dependence.png       # RF partial dependence (top-2 variables)
        ├── rf_importance.png            # RF %IncMSE variable importance (top-10)
        ├── cox_schoenfeld.png           # Cox PH Schoenfeld residual plots
        ├── irf_sem_to_drift.png         # VAR impulse-response (90% CI bands)
        ├── gam_smooth_plots.png         # GAM smooth effects + te() interaction
        ├── coef_comparison.png          # β across OLS / FE / SAR / IV specs
        ├── spatial_scatter.png          # District sem-dist vs drift-stability scatter
        ├── temporal_evolution.png       # 17-quarter time series (3 metrics)
        ├── cohort_heatmap.png           # Cohort × Quarter mean semantic distance
        ├── tenure_effect.png            # Semantic distance & drift by tenure
        ├── residuals_hist.png           # OLS residual distribution with KDE
        ├── influence_plot.png           # Leverage vs Cook's D influence plot
        └── quantile_coef.png            # QR coefficient plot τ = 0.1–0.9
```

---

## R Robustness Analysis (`r_robustness_analysis/`)

The R analysis in this folder extends and complements the main Python pipeline with a suite of robustness, sensitivity, and validation tests that are not reported in the main analysis scripts. All analyses are run on the same simulated panel structure described in Section 3 of the paper.

### What this folder adds

| Enhancement | Description |
|---|---|
| **ST-01** | EFA with Horn parallel analysis + MAP (Velicer) test for construct dimensionality |
| **ST-02** | Within / Between / Overall R² decomposition (plm) |
| **ST-03** | Kleibergen-Paap rk F-statistic for weak instrument detection (fixest) |
| **ST-04** | Window-level Wald test for momentum coefficient homogeneity across 2Q–6Q windows |
| **ST-05** | Stacked ensemble meta-learner (Ridge combining EN, LASSO, Ridge, RF, GBM) |
| **ST-06** | Cox PH Schoenfeld residual plots (auto-saved to PNG) |
| **ST-07** | VECM estimation, conditional on Johansen cointegration rank ≥ 1 |
| **ST-08** | GAM with tensor-product te() interaction (SEMANTIC_DISTANCE × OSCILLATION) |
| **ST-09** | BCa double-bootstrap confidence intervals (boot package) |
| **ST-10** | Cook's D influence diagnostics with leverage-Cook scatter plot |
| **VZ-01** | Cohort × Quarter heatmap of mean semantic distance |
| **VZ-02** | IRF with 90% bootstrap confidence bands |
| **VZ-03** | GAM smooth-effect plots (all terms including te()) |
| **VZ-04** | Unified `theme_drift()` function applied to all ggplot outputs |

### Key design choices

- **Two-pass lag construction** (BUG-STEP fix): `pos_step` and `neg_step` are computed in two explicit `mutate()` passes rather than `lag(default = first(...))`, which silently returns all-zero columns in dplyr ≥ 1.1.
- **Matrix indexing** (BUG-RW fix): the random-walk panel uses `rw_mat[cbind(listing_id, quarter_id)]` to avoid row-recycling artefacts.
- **VIF type guard** (BUG-VIF fix): `car::vif()` output is coerced to a named numeric vector before printing, handling environments where it returns a matrix.
- **MICE m = 5**: five imputation datasets with Rubin's rules pool check reported to `mice_pool_check.csv`.
- **irlba PCA + UMAP**: 768-d → 32-d reduction using truncated SVD; UMAP-32 sensitivity check reports cosine-distance fidelity relative to PCA-32 (r = 0.972).

### Running the R analysis

```r
# Install dependencies (first run only)
# The script auto-installs missing packages from CRAN.

# Set working directory to r_robustness_analysis/
setwd("r_robustness_analysis")

# Run the full pipeline (~3–5 min on M-series / x86-64 with 16 GB RAM)
source("semantic_drift_robustness_v6.R")
```

All CSV and PNG outputs are written to the working directory. A timestamped log is saved to `semantic_drift_v6.log`.

### R dependencies

```r
required_pkgs <- c(
  "tidyverse", "lubridate", "zoo", "truncnorm", "mice",
  "plm", "fixest", "lmtest", "sandwich", "AER",
  "gmm", "sampleSelection", "MatchIt", "survival",
  "glmnet", "randomForest", "caret", "gbm",
  "mgcv", "quantreg", "boot", "broom",
  "car", "tseries", "nortest", "strucchange",
  "forecast", "vars", "psych", "ggrepel",
  "uwot", "proxy", "FactoMineR", "irlba",
  "purrr", "tibble", "scales", "tsDyn",
  "urca", "mFilter", "patchwork", "viridis"
)
```

---

## Data

### Source

Panel data were assembled from [Inside Airbnb](http://insideairbnb.com/) quarterly snapshots for Hong Kong. The raw data are publicly available and can be accessed directly from the Inside Airbnb website.

### Preprocessed Panel (`data/IV_panel.csv`)

The file `IV_panel.csv` contains the integrated listing × quarter panel after an eight-stage preprocessing pipeline (described in Section 3.1 of the paper). Each row represents one listing in one quarter. Key variables are described below.

| Variable | Type | Description |
|---|---|---|
| `listing_id` | int | Unique listing identifier |
| `period_qtr` | str | Fiscal quarter (e.g., `2021Q1`) |
| `neighbourhood_cleansed` | str | Hong Kong administrative district (18 districts) |
| `sem_distance` | float | Cosine distance between listing embedding and district-quarter centroid (Semantic Discrepancy Index, SDI) |
| `sem_std` | float | Within-quarter standard deviation of review-level cosine distances (representational variability proxy) |
| `delta_sem_distance` | float | Quarter-on-quarter change in SDI (Δsem_distance) |
| `n_reviews_qtr` | int | Number of guest reviews in the quarter |
| `price_log` | float | Log-transformed nightly price (USD) |
| `superhost_flag` | int | Binary indicator: 1 = Airbnb Superhost status |
| `amenity_count` | int | Number of listed amenities |
| `sentiment_mean_qtr` | float | Mean quarterly VADER compound sentiment score |

> **Note on data:** All results in the current version of the codebase are based on a simulated embedding pipeline that replicates the structure of the Hong Kong panel. Applying the pipeline to actual Inside Airbnb listing text is identified as the primary next step in the paper's limitations section.

---

## Methods

### Semantic Embedding Pipeline

Listing descriptions and aggregated quarterly review corpora are encoded using **Sentence-BERT** (`all-MiniLM-L6-v2`), a transformer-based model optimized for semantic similarity. Embeddings are reduced from 768 to 32 dimensions via irlba-based truncated SVD (PCA), with a UMAP-32 sensitivity analysis confirming fidelity (cosine-distance correlation r = 0.972). The **Semantic Discrepancy Index (SDI)** is the cosine distance between a listing's supply-side embedding and the period-specific, district-level demand centroid:

```
SDI(i,t) = 1 − cos(s_i, r_{i,t})
```

where `s_i` is the listing embedding and `r_{i,t}` is the district-quarter demand centroid. Values range from 0 (perfect alignment) to 2 (complete opposition).

### Representational Momentum

Quarter-on-quarter changes in SDI are decomposed into directionally signed components:

```
pos_step(i,t) = max(−ΔSDI(i,t), 0)   # convergence toward centroid
neg_step(i,t) = max(+ΔSDI(i,t), 0)   # divergence from centroid
```

Four-quarter rolling sums (`positive_momentum_4q`, `negative_momentum_4q`) aggregate directional tendency across a multi-period window. Window sensitivity is assessed across 2Q–6Q horizons with Cochran Q heterogeneity tests (reported in `window_sensitivity_results.csv`) and Wald coefficient-homogeneity tests (ST-04).

### Estimation Strategy

| Hypothesis | Model | Specification | Standard Errors |
|---|---|---|---|
| H1a, H1b | OLS (M1–M3) | Lifetime ~ Fit + Variability + Controls + Entry-Quarter FE | HC1 heteroscedasticity-robust |
| H2a, H2b | Discrete-time logit (M4–M6) | Exit ~ Momentum (lagged) + Controls + Quarter FE | Clustered at listing level |
| H3a, H3b | OLS subgroup (Ext-A/B/C) | Same as H1 ± Interaction terms | HC1 heteroscedasticity-robust |
| Spatial diagnostic | Spatial Lag / SAR | Drift Stability ~ ρ·W·y + X·β + ε | Maximum likelihood |

Spatial autocorrelation is assessed via Global Moran's I on OLS residuals. District-level spatial dependence is modeled using a K-nearest neighbors (K = 5) weight matrix over Hong Kong's 18 administrative district centroids.

---

## Installation

### Python (main pipeline)

```bash
Python >= 3.9
pip install pandas numpy statsmodels lifelines matplotlib
```

For semantic embedding (upstream preprocessing):

```bash
pip install sentence-transformers scikit-learn vaderSentiment
```

For spatial econometrics (SAR/Moran's I):

```bash
pip install pysal libpysal spreg esda
```

### R (robustness analysis)

```bash
R >= 4.2.0
# All packages are auto-installed on first run via install.packages()
```

---

## Reproducing the Results

### Step 1 — Clone the repository

```bash
git clone https://github.com/LEEYJ1021/platform-representational-survival.git
cd platform-representational-survival
```

### Step 2 — Update data path

Open `main_analysis.py` and `extension_analysis.py` and set `PANEL_PATH` to the location of `IV_panel.csv` on your system:

```python
PANEL_PATH = "data/IV_panel.csv"
```

### Step 3 — Run H1 and H2 analyses

```bash
python main_analysis.py
```

**Outputs:** Full regression tables (M1–M3 for H1; M4–M6 for H2), VIF diagnostics, momentum direction summary, CSV result files.

### Step 4 — Run spatial subgroup extension (H3)

```bash
python extension_analysis.py
```

**Outputs:** Ext-A through Ext-C regression tables, slope comparison summary, CSV files, `extension_spatial_plots.png`.

### Step 5 — Run R robustness analysis

```bash
cd r_robustness_analysis
Rscript semantic_drift_robustness_v6.R
```

**Outputs:** All CSV and PNG files listed in the repository structure above, plus `semantic_drift_v6.RData` (full workspace) and `semantic_drift_v6.log` (timestamped execution log).

### Expected Runtime

| Script | Hardware | Time |
|---|---|---|
| `main_analysis.py` | Apple M-series / x86-64, 16 GB RAM | < 2 min |
| `extension_analysis.py` | Apple M-series / x86-64, 16 GB RAM | < 2 min |
| `semantic_drift_robustness_v6.R` | Apple M-series / x86-64, 16 GB RAM | < 5 min |

---

## Results Summary

### H1: Representational Fit, Variability, and Listing Lifetime

| Model | Fit (H1a) | Variability (H1b) | N | R² | Adj. R² |
|---|---|---|---|---|---|
| M1: Baseline | −0.928† | −0.869 (n.s.) | 317 | 0.240 | 0.191 |
| M2: Fit only | −0.931† | — | 317 | 0.239 | 0.193 |
| M3: Variability only | — | −0.888 (n.s.) | 317 | 0.234 | 0.188 |

†*p* < 0.10. HC1 robust SE. Entry-quarter fixed effects included.

### H2: Representational Momentum and Exit Risk

| Model | Momentum | β | OR | *p* |
|---|---|---|---|---|
| M4 (H2a) | Positive, 4Q cumulative | −0.290 | **0.75** | < 0.10 |
| M5 (H2b) | Negative, 4Q cumulative | +0.246 | **1.28** | < 0.10 |
| M4 (H2a, contemporaneous) | Positive, single-period | −0.105 | 0.90 | n.s. |
| M5 (H2b, contemporaneous) | Negative, single-period | +0.048 | 1.05 | n.s. |

Listing-clustered SE. Quarter fixed effects included. N = 3,407 listing-quarters; exit rate = 13.9%.

### H3: Spatial Subgroup Extension

| Construct | Tourist-Core (Ext-A) | Non-Tourist (Ext-B) | Direction |
|---|---|---|---|
| Representational fit | −0.840 (n.s.) | −0.790 (n.s.) | Same |
| Representational variability | +3.030* | −0.850 (n.s.) | ★ Reversed |

\**p* < 0.05 (tourist-core variability only). All other subgroup estimates exploratory. Tourist-core = Yau Tsim Mong, Wan Chai, Central & Western (n = 188). Non-tourist = remaining 15 districts (n = 129).

> **Note:** All results are based on a simulated embedding pipeline rather than actual Airbnb listing text, and should be treated as directional and exploratory rather than empirically definitive.

---

## Acknowledgments

Panel data are drawn from [Inside Airbnb](http://insideairbnb.com/), an independent, non-commercial project. Semantic embeddings use the `all-MiniLM-L6-v2` model from [Sentence-Transformers](https://www.sbert.net/) (Reimers & Gurevych, 2019). Spatial weight matrices are constructed using [PySAL](https://pysal.org/). R robustness analyses use [irlba](https://cran.r-project.org/package=irlba) for truncated SVD, [fixest](https://cran.r-project.org/package=fixest) for high-dimensional fixed effects, and [uwot](https://cran.r-project.org/package=uwot) for UMAP.

---

## License

This project is released under the [MIT License](LICENSE). The underlying Inside Airbnb data are subject to their own [terms of use](http://insideairbnb.com/about/).

---

## Contact

For questions about the code or data, please open an issue on this repository or contact the author directly via GitHub.

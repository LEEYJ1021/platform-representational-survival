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

This repository contains all analysis code, preprocessed panel data, and result outputs needed to reproduce the empirical findings reported in the paper. The study examines how the **temporal structure** and **spatial context** of representational reconfiguration jointly determine listing survival on a digital accommodation platform.

Drawing on **dynamic capabilities theory** (Teece, 2007), the study operationalizes three temporally distinct constructs — *representational fit*, *variability*, and *momentum* — and tests their survival consequences on a **seventeen-quarter panel of 401 Airbnb listings** across 18 Hong Kong administrative districts (2021Q1–2025Q2). The analytical pipeline integrates Sentence-BERT semantic embedding, spatial lag (SAR) modeling, and discrete-time logit survival analysis, with a self-contained R module that replicates the core H1/H2/H3 estimates and produces all paper-facing visualizations.

**Key findings:**

- A spatial lag (SAR) model achieves an **in-sample R² = 0.808** on the 9 districts with complete semantic data, against near-zero performance for all non-spatial specifications — but leave-one-out cross-validation (R² = −25.993) confirms severe overfitting at this sample size, so the result is treated as **directional evidence** of competitive differentiation rather than imitative convergence.
- Representational fit is associated with shorter survival overall (β = −0.928, *p* = 0.088), and this overfit penalty is significant and substantially larger in **tourist-core districts** (β = −1.318, *p* = 0.049) than in non-tourist districts (β = +0.130, n.s.).
- Cumulative four-quarter representational convergence significantly **reduces exit odds by 29%** (OR = 0.714, *p* < .05); equivalent divergence is associated with only a small, non-significant **5% increase** in exit odds (OR = 1.050, n.s.) — an asymmetry confirmed by a Wald test (χ²(1) = 4.31, *p* = .038) and consistent with **gain-sensitivity** rather than loss aversion.
- Single-period (contemporaneous) momentum has **no detectable effect** on exit risk in either direction, supporting the paper's path-dependence argument: survival is governed by accumulated trajectory, not current position.

---

## Repository Structure

```
.
├── README.md
├── main_analysis.py                     # H1 (OLS lifetime) + H2 (discrete-time logit survival)
├── extension_analysis.py                # H3 spatial subgroup extension (tourist vs. non-tourist)
│
├── data/
│   ├── IV_panel.csv                     # Listing × quarter panel (preprocessed, 3,614 obs.) — Python pipeline input
│   └── dataset_0322.xlsx                # Two-sheet workbook (H12, H3) — R analysis input (see below)
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
└── r_analysis/                          # Self-contained R replication & visualization module
    ├── analysis_main.R                  # Full R pipeline: H1 OLS, H2 logit (Model A/B), H3 subgroup, 8 figures
    └── figures/                         # Auto-created by the script itself — all PNGs land here
        ├── fig1_h1_forest_plot.png
        ├── fig2_model_AB_comparison.png
        ├── fig3_beta_magnitude.png
        ├── fig4_temporal_evolution.png
        ├── fig5_spatial_subgroup.png
        ├── fig6_district_coverage.png
        ├── fig7_exit_rate_time.png
        └── fig8_odds_ratio_summary.png
```

> **Note on consolidation:** the R code and every figure it produces live inside the single `r_analysis/` folder. The script computes its own location at runtime (`script_dir`) and writes all outputs to `script_dir/figures`, so there is no separate top-level `figures/` or `csv_outputs/` directory to keep in sync — copying or moving `r_analysis/` as a unit is sufficient to carry the full R deliverable.

---

## R Analysis Module (`r_analysis/`)

`analysis_main.R` is a single, self-contained script that reads the raw H1/H2/H3 analytical samples directly from `data/dataset_0322.xlsx`, re-estimates the paper's core models in R, and saves all figures as PNGs into `r_analysis/figures/`. It is intended as a transparent, independently-runnable cross-check of the Python pipeline's H1/H2/H3 results, paired with the publication-quality visualizations used in the paper.

### What the script does

| Section | Description | Paper reference |
|---|---|---|
| **1. Data loading** | Reads sheets `H12` and `H3` from `dataset_0322.xlsx`; reconstructs the H1 listing-level sample and the H2 quarterly panel using the exact sequential exclusion filters described in the paper | Section 3.5.3 |
| **2. H1 OLS models** | Fits M1 (joint Fit + Variability + controls) with HC1 robust SE; reports H1a/H1b coefficients | Table 4 |
| **3. H2 discrete-time logit** | Fits Model A (contemporaneous pos_step/neg_step) and Model B (4Q cumulative, z-scored momentum) with manually computed listing-clustered SE; runs the Wald test for recovery-vs-deterioration magnitude asymmetry | Table 5, Section 4.4 |
| **4. Temporal evolution** | Aggregates mean semantic distance and exit rate by quarter across the full 2021Q1–2025Q2 panel | Figure 4 |
| **5. H3 spatial subgroup** | Re-estimates the H1 specification separately for tourist-core (Yau Tsim Mong, Wan Chai, Central & Western) and non-tourist districts | Table 6 |
| **6. Supplementary figures** | District-level data coverage and quarterly exit-rate trend | — |

### Figures produced

| File | Content |
|---|---|
| `fig1_h1_forest_plot.png` | H1 OLS coefficient forest plot (90%/95% CI), fit vs. variability vs. controls |
| `fig2_model_AB_comparison.png` | Model A (contemporaneous) vs. Model B (4Q cumulative) β and OR comparison — the path-dependence test |
| `fig3_beta_magnitude.png` | Recovery vs. deterioration β magnitude comparison with the Wald asymmetry statistic annotated |
| `fig4_temporal_evolution.png` | Market-level mean semantic distance, 2021Q1–2025Q2 |
| `fig5_spatial_subgroup.png` | H3 tourist-core vs. non-tourist vs. full-sample coefficient comparison (fit and variability) |
| `fig6_district_coverage.png` | Listing-quarter observation counts by administrative district, tourist-core highlighted |
| `fig7_exit_rate_time.png` | Quarterly exit rate over the panel period |
| `fig8_odds_ratio_summary.png` | Odds ratios for recovery/deterioration under Model A vs. Model B |

### Data requirements

The script expects `data/dataset_0322.xlsx` one directory above `r_analysis/` (i.e., at the repository root), containing two sheets:

| Sheet | Used for | Key columns |
|---|---|---|
| `H12` | H1 OLS listing-level sample | `listing_id`, `fit_init`, `variability_init`, `activity_init`, `reviews_init`, `price_init`, `superhost_init`, `amenity_init`, `sentiment_init`, `lifetime_quarters`, `entry_quarter` |
| `H3` | H2 logit panel + H3 subgroup + descriptive figures | `listing_id`, `period_qtr`, `neighbourhood_cleansed`, `new_spell`, `exit_next`, `pos_step`, `neg_step`, `positive_momentum_4q`, `negative_momentum_4q`, `platform_activity_lag`, `n_reviews_qtr_lag`, `sentiment_mean_qtr_lag`, `sem_distance` |

`dataset_0322.xlsx` is a separate, sheet-based artifact from `IV_panel.csv`: `IV_panel.csv` feeds the Python pipeline's full listing-quarter panel, while `H12`/`H3` are the pre-aggregated, model-ready samples (early-window listing averages and the filtered quarterly hazard panel, respectively) consumed directly by the R script.

### R dependencies

```r
required_pkgs <- c(
  "readxl", "dplyr", "tidyr", "ggplot2", "ggrepel",
  "sandwich", "lmtest", "forcats", "scales",
  "patchwork", "RColorBrewer", "tibble", "stringr"
)
```

Missing packages are auto-installed from CRAN the first time the script runs.

### Running the R analysis

```r
# Set working directory to r_analysis/
setwd("r_analysis")

# Run the full pipeline (well under 1 min on a typical laptop)
source("analysis_main.R")
```

Console output reports, in order: sample sizes for the H1 and H2 estimation samples, the M1 R² and H1a/H1b coefficients, the Model B odds ratios for positive and negative 4Q momentum, the Wald asymmetry test result, the tourist-core/non-tourist subgroup coefficients, and a manifest of every PNG written to `figures/`.

### Key design choices

- **Manual cluster-robust SE**: standard errors for the H2 discrete-time logit Model B are clustered at `listing_id` via a hand-written sandwich-estimator function (`cluster_vcov()`), rather than an external package, reproducing the listing-clustered SE reported in Table 5 of the paper.
- **Standardized momentum**: `positive_momentum_4q` and `negative_momentum_4q` are z-scored within the estimation sample before entering Model B, per Section 3.3.2 of the paper, so that the Wald test compares directly comparable magnitudes.
- **Exact sample reconstruction**: the H2 panel is rebuilt using the same sequential filters as the paper — drop first-quarter spells (`new_spell == 0`) → drop rows with missing 4Q momentum → drop the terminal quarter → drop zero-event quarters → drop rows with missing lagged controls.
- **90%/95% dual-CI forest plot**: Figure 1 displays both confidence levels and codes significance at the 90% threshold, matching the paper's convention of reporting marginal significance (e.g., β = −0.928†, *p* = 0.088) for the H1a fit effect.

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

### R Analysis Input (`data/dataset_0322.xlsx`)

See [R Analysis Module](#r-analysis-module-r_analysis) above for the full sheet/column specification (`H12`, `H3`).

> **Note on data:** All results in the current version of the codebase are based on a simulated embedding pipeline that replicates the structure of the Hong Kong panel. Applying the pipeline to actual Inside Airbnb listing text is identified as the primary next step in the paper's limitations section.

---

## Methods

### Semantic Embedding Pipeline

Listing descriptions and aggregated quarterly review corpora are encoded using **Sentence-BERT** (`all-MiniLM-L6-v2`), a transformer-based model optimized for semantic similarity. Embeddings are reduced from 768 to 32 dimensions via PCA, retaining 95% of original variance. The **Semantic Discrepancy Index (SDI)** is the cosine distance between a listing's supply-side embedding and the period-specific, district-level demand centroid:

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

Four-quarter rolling sums (`positive_momentum_4q`, `negative_momentum_4q`) aggregate directional tendency across a multi-period window, capturing the purposive trajectory central to the dynamic capabilities account of reconfiguration.

### Estimation Strategy

| Hypothesis | Model | Specification | Standard Errors |
|---|---|---|---|
| H1a, H1b | OLS (M1–M3) | Lifetime ~ Fit + Variability + Controls + Entry-Quarter FE | HC1 heteroscedasticity-robust |
| H2a, H2b | Discrete-time logit (Model A / Model B) | Exit ~ Momentum (contemporaneous / 4Q cumulative, lagged) + Controls + Quarter FE | Clustered at listing level |
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

### R (analysis & visualization module)

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

For the R module, ensure `dataset_0322.xlsx` is placed at `data/dataset_0322.xlsx` at the repository root — `r_analysis/analysis_main.R` resolves this path automatically as `../data/dataset_0322.xlsx` relative to its own location.

### Step 3 — Run H1 and H2 analyses (Python)

```bash
python main_analysis.py
```

**Outputs:** Full regression tables (M1–M3 for H1; M4–M6 for H2), VIF diagnostics, momentum direction summary, CSV result files.

### Step 4 — Run spatial subgroup extension (Python, H3)

```bash
python extension_analysis.py
```

**Outputs:** Ext-A through Ext-C regression tables, slope comparison summary, CSV files, `extension_spatial_plots.png`.

### Step 5 — Run R analysis & visualization module

```bash
cd r_analysis
Rscript analysis_main.R
```

**Outputs:** 8 PNG figures replicating and visualizing the H1 (Table 4), H2 (Table 5), and H3 (Table 6) results, written directly to `r_analysis/figures/` — no separate output directory needed.

### Expected Runtime

| Script | Hardware | Time |
|---|---|---|
| `main_analysis.py` | Apple M-series / x86-64, 16 GB RAM | < 2 min |
| `extension_analysis.py` | Apple M-series / x86-64, 16 GB RAM | < 2 min |
| `r_analysis/analysis_main.R` | Apple M-series / x86-64, 16 GB RAM | < 1 min |

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
| Model A (contemporaneous) | Recovery (pos_step) | −0.105 | 0.90 | n.s. |
| Model A (contemporaneous) | Deterioration (neg_step) | +0.048 | 1.05 | n.s. |
| Model B (4Q cumulative, H2a) | Cumulative recovery | −0.338 | **0.714** | < .05 |
| Model B (4Q cumulative, H2b) | Cumulative deterioration | +0.049 | 1.050 | n.s. |

Listing-clustered SE. Quarter fixed effects included. N = 1,798 listing-quarters; per-quarter exit rate = 6.7%. Wald asymmetry test: χ²(1) = 4.31, *p* = .038 (recovery dominates).

### H3: Spatial Subgroup Extension

| Construct | Tourist-Core (Ext-A) | Non-Tourist (Ext-B) | Direction |
|---|---|---|---|
| Representational fit | −1.318* | +0.130 (n.s.) | ★ Reversed |
| Representational variability | +1.725 (n.s.) | −4.474† | ★ Reversed |

\**p* < 0.05, †*p* < 0.10. Tourist-core = Yau Tsim Mong, Wan Chai, Central & Western (n = 188). Non-tourist = remaining 15 districts (n = 129).

> **Note:** All results are based on a simulated embedding pipeline rather than actual Airbnb listing text, and should be treated as directional and exploratory rather than empirically definitive.

---

## Acknowledgments

Panel data are drawn from [Inside Airbnb](http://insideairbnb.com/), an independent, non-commercial project. Semantic embeddings use the `all-MiniLM-L6-v2` model from [Sentence-Transformers](https://www.sbert.net/) (Reimers & Gurevych, 2019). Spatial weight matrices are constructed using [PySAL](https://pysal.org/). The R analysis and visualization module uses [ggplot2](https://ggplot2.tidyverse.org/) and [patchwork](https://patchwork.data-imaginist.com/) for figure composition, and [sandwich](https://cran.r-project.org/package=sandwich) / [lmtest](https://cran.r-project.org/package=lmtest) for robust inference.

---

## License

This project is released under the [MIT License](LICENSE). The underlying Inside Airbnb data are subject to their own [terms of use](http://insideairbnb.com/about/).

---

## Contact

For questions about the code or data, please open an issue on this repository or contact the author directly via GitHub.

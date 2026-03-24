# Conformity, Trajectory, and Space: A Dynamic Capabilities Account of Representational Survival in Platform Accommodation Markets

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data: Inside Airbnb](https://img.shields.io/badge/Data-Inside%20Airbnb-orange.svg)](http://insideairbnb.com/)
[![Framework: Statsmodels](https://img.shields.io/badge/Framework-Statsmodels-green.svg)](https://www.statsmodels.org/)

> **Replication materials for the paper:**
> *Conformity, Trajectory, and Space: A Dynamic Capabilities Account of Representational Survival in Platform Accommodation Markets*

---

## Overview

This repository contains all analysis code, preprocessed panel data, and result outputs to fully reproduce the empirical findings reported in the paper. The study examines how the **temporal structure** and **spatial context** of representational reconfiguration jointly determine listing survival on a digital accommodation platform.

Drawing on **dynamic capabilities theory** (Teece, 2007), the study operationalizes three temporally distinct constructs—*representational fit*, *variability*, and *momentum*—and tests their survival consequences on a **seventeen-quarter panel of 401 Airbnb listings** across 18 Hong Kong administrative districts (2021Q1–2025Q2). The analytical pipeline integrates Sentence-BERT semantic embedding, spatial lag (SAR) modeling, and discrete-time logit survival analysis.

**Key findings:**

- A spatial lag (SAR) model recovers **R² = 28.3%** of variance in semantic drift stability, against near-zero performance for all non-spatial specifications, indicating that competitive differentiation—not imitative convergence—is the dominant coordination mechanism.
- High representational fit **shortens survival** in tourist-core districts (β = −1.318, *p* < .05) while carrying no survival cost in stable residential districts—two structurally opposite logics within the same platform.
- Cumulative four-quarter convergence reduces exit odds by **25%** (OR = 0.75); equivalent divergence increases them by **28%** (OR = 1.28). Neither effect appears in single-period specifications, confirming a path-dependent trajectory mechanism consistent with supply-side loss aversion.

---

## Repository Structure

```
.
├── README.md
├── main_analysis.py            # H1 (OLS lifetime) + H2 (discrete-time logit survival)
├── extension_analysis.py       # H3 spatial subgroup extension (tourist vs. non-tourist)
│
├── data/
│   └── IV_panel.csv            # Listing × quarter panel (preprocessed, 3,614 obs.)
│
└── results/
    ├── results_H1_M1_baseline.csv          # H1 Model M1: Fit + Variability (joint)
    ├── results_H1_M2_fit.csv               # H1 Model M2: Fit only
    ├── results_H1_M3_variability.csv       # H1 Model M3: Variability only
    ├── results_H2_M4_positive_momentum.csv # H2 Model M4: Positive momentum
    ├── results_H2_M5_negative_momentum.csv # H2 Model M5: Negative momentum
    ├── results_H2_M6_joint.csv             # H2 Model M6: Joint momentum (both directions)
    ├── results_ExtA_tourist.csv            # H3 Ext-A: Tourist-core subgroup
    ├── results_ExtB_nontourist.csv         # H3 Ext-B: Non-tourist subgroup
    ├── results_ExtC_interaction.csv        # H3 Ext-C: Full-sample interaction model
    ├── data_H1_listing_level.csv           # H1 estimation sample (listing-level)
    ├── data_H2_panel_survival.csv          # H2 estimation sample (quarterly panel)
    └── extension_spatial_plots.png         # Figure: 6-panel spatial heterogeneity visualization
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

> **Note on privacy:** All data are sourced from publicly available Inside Airbnb snapshots. No personally identifiable information is included. Listing IDs match the Inside Airbnb naming convention.

---

## Methods

### Semantic Embedding Pipeline

Listing descriptions and aggregated quarterly review corpora are encoded using **Sentence-BERT** (`all-MiniLM-L6-v2`), a transformer-based model optimized for semantic similarity. Embeddings are reduced from 768 to 32 dimensions via PCA (retaining ≥95% variance). The **Semantic Discrepancy Index (SDI)** is the cosine distance between a listing's supply-side embedding and the period-specific, district-level demand centroid:

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
| H2a, H2b | Discrete-time logit (M4–M6) | Exit ~ Momentum (lagged) + Controls + Quarter FE | Clustered at listing level |
| H3a, H3b | OLS subgroup (Ext-A/B/C) | Same as H1 ± Interaction terms | HC1 heteroscedasticity-robust |
| Spatial diagnostic | Spatial Lag / SAR | Drift Stability ~ ρ·W·y + X·β + ε | Maximum likelihood |

Spatial autocorrelation is assessed via Global Moran's I on OLS residuals. District-level spatial dependence is modeled using a K-nearest neighbors (K = 5) weight matrix over Hong Kong's 18 administrative district centroids.

---

## Installation

### Requirements

```bash
Python >= 3.9
```

### Install Dependencies

```bash
pip install pandas numpy statsmodels lifelines matplotlib
```

For semantic embedding (upstream preprocessing, not required to reproduce paper results from `IV_panel.csv`):

```bash
pip install sentence-transformers scikit-learn vaderSentiment
```

For spatial econometrics (SAR/Moran's I):

```bash
pip install pysal libpysal spreg esda
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

**Outputs:**

- Console: Full regression tables (M1–M3 for H1; M4–M6 for H2), VIF diagnostics, momentum direction summary
- CSV: `results_H1_M1_baseline.csv` through `results_H2_M6_joint.csv`, `data_H1_listing_level.csv`, `data_H2_panel_survival.csv`

### Step 4 — Run spatial subgroup extension (H3)

```bash
python extension_analysis.py
```

**Outputs:**

- Console: Ext-A (tourist-core), Ext-B (non-tourist), and Ext-C (interaction) regression tables with slope comparison summary
- CSV: `results_ExtA_tourist.csv`, `results_ExtB_nontourist.csv`, `results_ExtC_interaction.csv`
- Figure: `extension_spatial_plots.png` — 6-panel visualization of spatial heterogeneity in representational dynamics

### Expected Runtime

Both scripts complete in under **2 minutes** on a standard laptop (Apple M-series or equivalent x86-64 with 16 GB RAM).

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
| Representational fit | −1.318* | +0.130 (n.s.) | ★ Reversed |
| Representational variability | +1.725 (n.s.) | −4.474† | ★ Reversed |

*p* < 0.05; †*p* < 0.10. Tourist-core = Yau Tsim Mong, Wan Chai, Central & Western (n = 188). Non-tourist = remaining 15 districts (n = 129).

---

## Acknowledgments

Panel data are drawn from [Inside Airbnb](http://insideairbnb.com/), an independent, non-commercial project. Semantic embeddings use the `all-MiniLM-L6-v2` model from [Sentence-Transformers](https://www.sbert.net/) (Reimers & Gurevych, 2019). Spatial weight matrices are constructed using [PySAL](https://pysal.org/).

---

## License

This project is released under the [MIT License](LICENSE). The underlying Inside Airbnb data are subject to their own [terms of use](http://insideairbnb.com/about/).

---

## Contact

For questions about the code or data, please open an issue on this repository or contact the author directly via GitHub.

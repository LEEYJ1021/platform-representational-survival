"""
extension_analysis.py
=====================
Extension: 관광 핵심 구 vs 비관광 구에 따른
           Fit / Variability → Lifetime 효과의 공간적 이질성

근거:
  - Yau Tsim Mong, Wan Chai, Central & Western 세 구에
    전체 listing의 86% 집중 (Martin Yu, 2020; Jiang et al., 2024)
  - 이 공간적 맥락이 representational dynamics를 조건짓는가?

출력:
  - Ext-A: Tourist-core 서브샘플 OLS
  - Ext-B: Non-tourist 서브샘플 OLS
  - Ext-C: 전체 샘플 상호작용 모형 (기울기 차이 통계 검증)
  - Figure: 6-panel 시각화
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

PANEL_PATH   = "/Users/tslee/Desktop/platformef/Rasing_Valley/data/IV_panel.csv"
EARLY_WINDOW = 4

# ── 관광 핵심 구 정의 ──────────────────────────────────────
TOURIST_CORE = {"Yau Tsim Mong", "Wan Chai", "Central & Western"}

# =========================================================
# 1. LOAD & PREP (main_analysis와 동일한 전처리)
# =========================================================
df = pd.read_csv(PANEL_PATH)
df = df.loc[:, ~df.columns.duplicated()]
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
df = df.rename(columns={"Δsem_distance": "delta_sem_distance"})
df.columns = [c.strip() for c in df.columns]

df["quarter"]    = pd.PeriodIndex(df["period_qtr"].astype(str), freq="Q")
df["is_tourist"] = df["neighbourhood_cleansed"].isin(TOURIST_CORE).astype(int)

# LOO platform activity
regional = (
    df.groupby(["neighbourhood_cleansed", "quarter"])
      .agg(total_reviews=("n_reviews_qtr", "sum"))
      .reset_index()
)
df = df.merge(regional, on=["neighbourhood_cleansed", "quarter"], how="left")
df["platform_activity"] = np.log1p(df["total_reviews"] - df["n_reviews_qtr"])
df = df.drop(columns=["total_reviews"])

# First continuous spell
df = df.sort_values(["listing_id", "quarter"]).reset_index(drop=True)
df["prev_q"]    = df.groupby("listing_id")["quarter"].shift(1)
df["new_spell"] = (
    df["prev_q"].isna() | (df["quarter"] != df["prev_q"] + 1)
).astype(int)
df["spell_id"]  = df.groupby("listing_id")["new_spell"].cumsum()
panel = df[df["spell_id"] == 1].copy()

lifetime_df = (
    panel.groupby("listing_id")
         .agg(lifetime_quarters=("quarter", "count"),
              entry_quarter=("quarter", "min"))
         .reset_index()
)
lifetime_df["entry_quarter_str"] = lifetime_df["entry_quarter"].astype(str)

# =========================================================
# 2. INITIAL WINDOW AGGREGATION
# =========================================================
panel["order"] = panel.groupby("listing_id").cumcount() + 1
init = panel[panel["order"] <= EARLY_WINDOW].copy()

h_df = (
    init.groupby("listing_id")
        .agg(
            fit_init         = ("sem_distance",        "mean"),
            variability_init = ("sem_std",             "mean"),
            activity_init    = ("platform_activity",   "mean"),
            reviews_init     = ("n_reviews_qtr",       "mean"),
            price_init       = ("price_log",           "mean"),
            superhost_init   = ("superhost_flag",      "max"),
            amenity_init     = ("amenity_count",       "mean"),
            sentiment_init   = ("sentiment_mean_qtr",  "mean"),
            is_tourist       = ("is_tourist",          "first"),
        )
        .reset_index()
)
h_df = h_df.merge(lifetime_df, on="listing_id", how="left")

# 중심화 + 상호작용항
for v in ["fit_init", "variability_init", "activity_init"]:
    h_df[f"{v}_c"] = h_df[v] - h_df[v].mean()
h_df["tourist_c"]      = h_df["is_tourist"] - h_df["is_tourist"].mean()
h_df["fit_x_tourist"]  = h_df["fit_init_c"]         * h_df["tourist_c"]
h_df["var_x_tourist"]  = h_df["variability_init_c"]  * h_df["tourist_c"]

NEEDED = [
    "lifetime_quarters", "fit_init", "variability_init", "activity_init",
    "is_tourist", "fit_init_c", "variability_init_c", "tourist_c",
    "fit_x_tourist", "var_x_tourist",
    "reviews_init", "price_init", "superhost_init",
    "amenity_init", "sentiment_init", "entry_quarter_str"
]
h_clean = h_df.dropna(subset=NEEDED).copy()

tc_df  = h_clean[h_clean["is_tourist"] == 1].copy()
non_df = h_clean[h_clean["is_tourist"] == 0].copy()

print(f"전체: {len(h_clean)}  tourist-core: {len(tc_df)}  non-tourist: {len(non_df)}")

# =========================================================
# 3. MODELS
# =========================================================
CTRL = ("reviews_init + price_init + superhost_init + "
        "amenity_init + sentiment_init + C(entry_quarter_str)")

f_base    = f"lifetime_quarters ~ fit_init + variability_init + activity_init + {CTRL}"
f_interact= (
    f"lifetime_quarters ~ fit_init_c + variability_init_c + tourist_c + "
    f"fit_x_tourist + var_x_tourist + {CTRL}"
)

m_tc      = smf.ols(f_base,    data=tc_df  ).fit(cov_type="HC1")
m_non     = smf.ols(f_base,    data=non_df ).fit(cov_type="HC1")
m_interact= smf.ols(f_interact, data=h_clean).fit(cov_type="HC1")

# =========================================================
# 4. PRINT RESULTS
# =========================================================
for label, m, n in [
    ("Ext-A: Tourist-Core",    m_tc,      len(tc_df)),
    ("Ext-B: Non-Tourist",     m_non,     len(non_df)),
    ("Ext-C: Interaction",     m_interact,len(h_clean)),
]:
    print(f"\n{'='*55}")
    print(f"{label}  (n={n}, R²={m.rsquared:.3f}, Adj.R²={m.rsquared_adj:.3f})")
    print(f"{'='*55}")
    print(m.summary())

# ── Slope comparison table ──────────────────────────────
print(f"\n{'='*55}")
print("SLOPE COMPARISON SUMMARY")
print(f"{'='*55}")
print(f"{'변수':<22} {'Tourist β':>10} {'p':>7}  {'Non-Tourist β':>14} {'p':>7}")
print("-"*65)
for var in ["fit_init","variability_init","activity_init","reviews_init","price_init","superhost_init"]:
    bt, pt   = m_tc.params.get(var,np.nan),  m_tc.pvalues.get(var,np.nan)
    bn, pn   = m_non.params.get(var,np.nan), m_non.pvalues.get(var,np.nan)
    st = "***" if pt<0.001 else "**" if pt<0.01 else "*" if pt<0.05 else "†" if pt<0.10 else ""
    sn = "***" if pn<0.001 else "**" if pn<0.01 else "*" if pn<0.05 else "†" if pn<0.10 else ""
    print(f"{var:<22} {bt:>8.3f}{st:<3} {pt:>6.3f}  {bn:>12.3f}{sn:<3} {pn:>6.3f}")

print()
print(f"R² tourist={m_tc.rsquared:.3f} / non={m_non.rsquared:.3f} / full={m_interact.rsquared:.3f}")
print()
print("Interaction terms:")
for v in ["fit_x_tourist","var_x_tourist","tourist_c"]:
    b = m_interact.params.get(v,np.nan)
    p = m_interact.pvalues.get(v,np.nan)
    s = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.10 else "n.s."
    print(f"  {v:<22}: β={b:.3f}  p={p:.3f}  {s}")

# =========================================================
# 5. VISUALIZATION
# =========================================================
plt.rcParams.update({
    "font.family":"sans-serif", "font.size":11,
    "axes.spines.top":False, "axes.spines.right":False,
    "axes.linewidth":0.8, "grid.alpha":0.3, "figure.dpi":150,
})
C_TC  = "#2166AC"
C_NON = "#D6604D"
ALF   = 0.20

fig = plt.figure(figsize=(14, 10))
gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

def scatter_reg(ax, df_t, df_n, xvar, xlabel, ylabel="Lifetime (quarters)", title=""):
    for df_s, col, lab in [(df_t,C_TC,"Tourist-core"),(df_n,C_NON,"Non-tourist")]:
        xy = df_s[[xvar,"lifetime_quarters"]].dropna()
        x, y = xy[xvar].values, xy["lifetime_quarters"].values
        ax.scatter(x, y, color=col, alpha=ALF, s=18, linewidths=0)
        X  = np.column_stack([np.ones(len(x)), x])
        b  = np.linalg.lstsq(X, y, rcond=None)[0]
        r  = y - X@b; s2 = r.var(); cov = np.linalg.inv(X.T@X)*s2
        xr = np.linspace(x.min(), x.max(), 100)
        Xr = np.column_stack([np.ones(100), xr]); yr = Xr@b
        se = np.sqrt((Xr@cov*Xr).sum(axis=1))
        ax.plot(xr, yr, color=col, lw=2, label=f"{lab} (β={b[1]:.2f})")
        ax.fill_between(xr, yr-1.96*se, yr+1.96*se, color=col, alpha=0.12)
    ax.set_xlabel(xlabel, fontsize=10); ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=9, framealpha=0); ax.yaxis.grid(True, linestyle="--")

# (a) fit → lifetime
scatter_reg(fig.add_subplot(gs[0,0]), tc_df, non_df,
            "fit_init","Representational Fit (cosine dist.)",
            title="(a)  Fit → Lifetime")

# (b) variability → lifetime  [핵심]
ax_b = fig.add_subplot(gs[0,1])
scatter_reg(ax_b, tc_df, non_df,
            "variability_init","Representational Variability (sem_std)",
            ylabel="",
            title="(b)  Variability → Lifetime  ★")
# 방향 역전 강조
ax_b.annotate("Direction reversal\n(tourist ↑ / non-tourist ↓)",
              xy=(0.55,0.12), xycoords="axes fraction",
              fontsize=8, color="gray",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

# (c) coefficient forest plot
ax_c = fig.add_subplot(gs[0,2])
vars_fp  = ["fit_init","variability_init","activity_init","reviews_init"]
labs_fp  = ["Fit","Variability","Platform\nActivity","Reviews"]
ypos     = np.arange(len(vars_fp))
for i,(var,lab) in enumerate(zip(vars_fp,labs_fp)):
    for j,(m,col,grp) in enumerate([(m_tc,C_TC,"Tourist"),(m_non,C_NON,"Non-tourist")]):
        b  = m.params.get(var,np.nan); se = m.bse.get(var,np.nan)
        ax_c.errorbar(b, i+(j-0.5)*0.3, xerr=1.96*se,
                      fmt="o", color=col, ms=6, capsize=4, lw=1.5,
                      label=grp if i==0 else "")
ax_c.axvline(0, color="black", lw=0.8, ls="--")
ax_c.set_yticks(ypos); ax_c.set_yticklabels(labs_fp, fontsize=10)
ax_c.set_xlabel("Coefficient (95% CI)", fontsize=10)
ax_c.set_title("(c)  Coefficient Comparison", fontsize=11, fontweight="bold", pad=8)
ax_c.legend(fontsize=9, framealpha=0); ax_c.xaxis.grid(True, ls="--")

# (d) fit × district type
scatter_reg(fig.add_subplot(gs[1,0]), tc_df, non_df,
            "fit_init","Representational Fit",
            title="(d)  Fit × District Type")

# (e) variability × district type  [핵심 반복]
scatter_reg(fig.add_subplot(gs[1,1]), tc_df, non_df,
            "variability_init","Representational Variability",
            ylabel="",
            title="(e)  Variability × District Type  ★")

# (f) R² bar
ax_f   = fig.add_subplot(gs[1,2])
labels = ["Tourist\n(Ext-A)","Non-tourist\n(Ext-B)","Full\n(Ext-C)"]
r2s    = [m_tc.rsquared, m_non.rsquared, m_interact.rsquared]
r2a    = [m_tc.rsquared_adj, m_non.rsquared_adj, m_interact.rsquared_adj]
cols   = [C_TC, C_NON, "gray"]
xs     = np.arange(3)
ax_f.bar(xs, r2s, color=cols, alpha=0.7, width=0.5)
ax_f.bar(xs, r2a, color=cols, alpha=1.0, width=0.5, hatch="///")
for x,r in zip(xs,r2s):
    ax_f.text(x, r+0.005, f"{r:.3f}", ha="center", va="bottom", fontsize=9)
ax_f.set_xticks(xs); ax_f.set_xticklabels(labels, fontsize=9)
ax_f.set_ylabel("R²", fontsize=10)
ax_f.set_title("(f)  Model Fit by Subsample", fontsize=11, fontweight="bold", pad=8)
ax_f.set_ylim(0, max(r2s)*1.25); ax_f.yaxis.grid(True, ls="--")
# legend
import matplotlib.patches as mpatches
ax_f.legend(handles=[
    mpatches.Patch(facecolor="gray", alpha=0.7, label="R²"),
    mpatches.Patch(facecolor="gray", alpha=1.0, hatch="///", label="Adj. R²")
], fontsize=8, framealpha=0)

fig.suptitle(
    "Spatial Heterogeneity Extension:\n"
    "Representational Dynamics in Tourist-Core vs. Non-Tourist Districts",
    fontsize=13, fontweight="bold", y=1.01
)
fig.savefig("extension_spatial_plots.png", dpi=150, bbox_inches="tight", facecolor="white")
print("\nSaved: extension_spatial_plots.png")
plt.show()

# =========================================================
# 6. SAVE RESULTS
# =========================================================
def save_coef(model, fname):
    pd.DataFrame({
        "coef": model.params, "se": model.bse, "pval": model.pvalues
    }).to_csv(fname, encoding="utf-8-sig")
    print(f"Saved: {fname}")

save_coef(m_tc,      "results_ExtA_tourist.csv")
save_coef(m_non,     "results_ExtB_nontourist.csv")
save_coef(m_interact,"results_ExtC_interaction.csv")
print("\n완료: extension_analysis.py")

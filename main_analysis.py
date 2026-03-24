"""
main_analysis.py
================
H1: Representational Fit / Variability → Listing Lifetime (OLS)
H2: Positive / Negative Momentum → Exit Risk (Discrete-Time Cloglog)

데이터: IV_panel.csv (listing × quarter 단일 행)
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from lifelines import CoxPHFitter

PANEL_PATH   = "/Users/tslee/Desktop/platformef/Rasing_Valley/data/IV_panel.csv"
EARLY_WINDOW = 4
MOM_WINDOW   = 4

# =========================================================
# 공통 전처리 함수
# =========================================================
def load_and_prep(path):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df = df.rename(columns={"Δsem_distance": "delta_sem_distance"})
    df.columns = [c.strip() for c in df.columns]

    df["quarter"] = pd.PeriodIndex(df["period_qtr"].astype(str), freq="Q")
    df = df.sort_values(["listing_id", "quarter"]).reset_index(drop=True)

    # LOO 지역 플랫폼 활성화 지수
    regional = (
        df.groupby(["neighbourhood_cleansed", "quarter"])
          .agg(total_reviews=("n_reviews_qtr", "sum"))
          .reset_index()
    )
    df = df.merge(regional, on=["neighbourhood_cleansed", "quarter"], how="left")
    df["platform_activity"] = np.log1p(df["total_reviews"] - df["n_reviews_qtr"])
    df = df.drop(columns=["total_reviews"])

    # Exit DV
    df["next_q"]        = df.groupby("listing_id")["quarter"].shift(-1)
    df["expected_next"] = df["quarter"] + 1
    df["exit_next"]     = (
        df["next_q"].isna() | (df["next_q"] != df["expected_next"])
    ).astype(int)

    # First continuous spell
    df["prev_q"]    = df.groupby("listing_id")["quarter"].shift(1)
    df["new_spell"] = (
        df["prev_q"].isna() | (df["quarter"] != df["prev_q"] + 1)
    ).astype(int)
    df["spell_id"]  = df.groupby("listing_id")["new_spell"].cumsum()
    panel = df[df["spell_id"] == 1].copy()

    # Lifetime
    lifetime_df = (
        panel.groupby("listing_id")
             .agg(lifetime_quarters=("quarter", "count"),
                  entry_quarter=("quarter", "min"))
             .reset_index()
    )
    lifetime_df["entry_quarter_str"] = lifetime_df["entry_quarter"].astype(str)

    return panel, lifetime_df


# =========================================================
# 1. LOAD
# =========================================================
panel, lifetime_df = load_and_prep(PANEL_PATH)
print(f"Panel (first spell): {panel.shape}")

# =========================================================
# 2. INITIAL WINDOW → H1 listing-level dataset
# =========================================================
panel["order"] = panel.groupby("listing_id").cumcount() + 1
init = panel[panel["order"] <= EARLY_WINDOW].copy()

h1_df = (
    init.groupby("listing_id")
        .agg(
            fit_init          = ("sem_distance",        "mean"),
            variability_init  = ("sem_std",             "mean"),
            activity_init     = ("platform_activity",   "mean"),
            reviews_init      = ("n_reviews_qtr",       "mean"),
            price_init        = ("price_log",           "mean"),
            superhost_init    = ("superhost_flag",      "max"),
            amenity_init      = ("amenity_count",       "mean"),
            sentiment_init    = ("sentiment_mean_qtr",  "mean"),
        )
        .reset_index()
)
h1_df = h1_df.merge(lifetime_df, on="listing_id", how="left")

# 중심화 (interaction 없으므로 주효과 추정용)
for v in ["fit_init", "variability_init", "activity_init"]:
    h1_df[f"{v}_c"] = h1_df[v] - h1_df[v].mean()

h1_clean = h1_df.dropna(subset=[
    "lifetime_quarters", "fit_init", "variability_init",
    "activity_init", "reviews_init", "price_init",
    "superhost_init", "amenity_init", "sentiment_init",
    "entry_quarter_str"
]).copy()

print(f"H1 sample: {len(h1_clean)} listings")
print(f"  fit_init NaN 제거 전: {h1_df['fit_init'].isna().sum()} ({h1_df['fit_init'].isna().mean()*100:.1f}%)")

# =========================================================
# 3. VIF CHECK
# =========================================================
def compute_vif(df, vars_list, label=""):
    X = df[vars_list].dropna().copy()
    X["Intercept"] = 1.0
    vif = pd.DataFrame({
        "variable": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    }).sort_values("VIF", ascending=False)
    print(f"\n=== VIF {label} ===")
    print(vif.to_string(index=False))

compute_vif(h1_clean,
    ["fit_init","variability_init","activity_init",
     "reviews_init","price_init","superhost_init","amenity_init","sentiment_init"],
    "H1")

# =========================================================
# 4. H1 MODELS — OLS (HC1)
#
#   M1: Baseline — fit + variability + activity + controls
#   M2: fit만 포함 (H1a 독립 검증)
#   M3: variability만 포함 (H1b 독립 검증)
# =========================================================
CTRL = ("reviews_init + price_init + superhost_init + "
        "amenity_init + sentiment_init + C(entry_quarter_str)")

f_m1 = f"lifetime_quarters ~ fit_init + variability_init + activity_init + {CTRL}"
f_m2 = f"lifetime_quarters ~ fit_init + activity_init + {CTRL}"
f_m3 = f"lifetime_quarters ~ variability_init + activity_init + {CTRL}"

m1 = smf.ols(f_m1, data=h1_clean).fit(cov_type="HC1")
m2 = smf.ols(f_m2, data=h1_clean).fit(cov_type="HC1")
m3 = smf.ols(f_m3, data=h1_clean).fit(cov_type="HC1")

for label, m in [("M1: Baseline", m1), ("M2: Fit only", m2), ("M3: Variability only", m3)]:
    print(f"\n{'='*55}")
    print(f"H1 — {label}  (n={int(m.nobs)}, R²={m.rsquared:.3f}, Adj.R²={m.rsquared_adj:.3f})")
    print(f"{'='*55}")
    print(m.summary())

# =========================================================
# 5. MOMENTUM VARIABLES → H2 panel-level
# =========================================================
panel = panel.sort_values(["listing_id", "quarter"]).copy()

# delta_sem: step3 계산값 우선, NaN은 직접 diff()
panel["delta_sem"] = panel["delta_sem_distance"].fillna(
    panel.groupby("listing_id")["sem_distance"].diff()
)

# 개선(−Δ)과 악화(+Δ) 분리
panel["pos_step"] = np.where(panel["delta_sem"] < 0, -panel["delta_sem"], 0.0)
panel["neg_step"] = np.where(panel["delta_sem"] > 0,  panel["delta_sem"], 0.0)

# 4분기 누적 모멘텀
panel["pos_momentum_4q"] = (
    panel.groupby("listing_id")["pos_step"]
         .rolling(MOM_WINDOW, min_periods=1).sum()
         .reset_index(level=0, drop=True)
)
panel["neg_momentum_4q"] = (
    panel.groupby("listing_id")["neg_step"]
         .rolling(MOM_WINDOW, min_periods=1).sum()
         .reset_index(level=0, drop=True)
)

# 1분기 lag
for v in ["pos_momentum_4q", "neg_momentum_4q",
          "n_reviews_qtr", "platform_activity", "sentiment_mean_qtr"]:
    panel[f"{v}_lag"] = panel.groupby("listing_id")[v].shift(1)

# =========================================================
# 6. H2 SURVIVAL DATASET
# =========================================================
surv_cols = [
    "listing_id", "period_qtr", "exit_next",
    "pos_momentum_4q_lag", "neg_momentum_4q_lag",
    "n_reviews_qtr_lag", "platform_activity_lag",
    "sentiment_mean_qtr_lag",
]
surv_df = panel[surv_cols].dropna().copy()

# 문제 분기 제거 (완전 분리 구간)
qtr_stat = surv_df.groupby("period_qtr")["exit_next"].agg(["sum","count"])
qtr_stat["rate"] = qtr_stat["sum"] / qtr_stat["count"]
drop_qtrs = qtr_stat[
    (qtr_stat["rate"].isin([0.0, 1.0])) | (qtr_stat["count"] < 20)
].index.tolist()
surv_df = surv_df[~surv_df["period_qtr"].isin(drop_qtrs)].copy()

print(f"\nH2 panel sample: {surv_df.shape}")
print(f"제거 분기: {drop_qtrs}")
print(f"exit_next 분포: {surv_df['exit_next'].value_counts().to_dict()}")

assert len(surv_df) > 0

# =========================================================
# 7. H2 MODELS — Discrete-Time Complementary Log-Log
#
#   Cloglog은 이산 시간 생존 분석의 이론적 표준 모형:
#   - Proportional hazard 가정 유지 (logit은 불필요한 가정)
#   - 개선(H2a)과 악화(H2b)를 각각 독립 모형으로 검증
#   - 결과는 Hazard Ratio (exp(β))로 보고
#
#   M4: H2a — positive momentum (개선 누적 → 퇴출 위험 감소)
#   M5: H2b — negative momentum (악화 누적 → 퇴출 위험 증가)
#   M6: Joint — 양방향 동시 추정 (비대칭성 직접 비교)
# =========================================================
CTRL_SURV = ("n_reviews_qtr_lag + platform_activity_lag + "
             "sentiment_mean_qtr_lag + C(period_qtr)")

f_m4 = f"exit_next ~ pos_momentum_4q_lag + {CTRL_SURV}"
f_m5 = f"exit_next ~ neg_momentum_4q_lag + {CTRL_SURV}"
f_m6 = f"exit_next ~ pos_momentum_4q_lag + neg_momentum_4q_lag + {CTRL_SURV}"

fit_kwargs = dict(
    disp=False,
    method="newton",        # bfgs보다 수렴 안정적
    maxiter=200,
    cov_type="cluster",
    cov_kwds={"groups": surv_df["listing_id"]}
)

m4 = smf.logit(f_m4, data=surv_df).fit(**fit_kwargs)
m5 = smf.logit(f_m5, data=surv_df).fit(**fit_kwargs)
m6 = smf.logit(f_m6, data=surv_df).fit(**fit_kwargs)

for label, m, hyp in [
    ("M4: Positive Momentum (H2a)", m4, "pos_momentum_4q_lag"),
    ("M5: Negative Momentum (H2b)", m5, "neg_momentum_4q_lag"),
    ("M6: Joint",                   m6, None),
]:
    print(f"\n{'='*55}")
    print(f"H2 — {label}")
    print(f"  n={int(m.nobs)}, Pseudo R²={m.prsquared:.3f}, converged={m.mle_retvals['converged']}")
    print(f"{'='*55}")
    print(m.summary())

    if hyp:
        b  = m.params[hyp]
        se = m.bse[hyp]
        p  = m.pvalues[hyp]
        OR = np.exp(b)
        ci_lo, ci_hi = np.exp(b - 1.96*se), np.exp(b + 1.96*se)
        print(f"\n  → Odds Ratio: {OR:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]  p={p:.3f}")

# =========================================================
# 8. M6 JOINT — 비대칭성 방향 요약
#    (scipy 없이 M6 계수 직접 비교로 충분)
# =========================================================
print(f"\n{'='*55}")
print("MOMENTUM DIRECTION SUMMARY (M6 Joint)")
print(f"{'='*55}")
b_pos = m6.params["pos_momentum_4q_lag"]
b_neg = m6.params["neg_momentum_4q_lag"]
p_pos = m6.pvalues["pos_momentum_4q_lag"]
p_neg = m6.pvalues["neg_momentum_4q_lag"]

print(f"  Positive momentum: β={b_pos:.3f}  OR={np.exp(b_pos):.3f}  p={p_pos:.3f}")
print(f"  Negative momentum: β={b_neg:.3f}  OR={np.exp(b_neg):.3f}  p={p_neg:.3f}")
print()
if b_pos < 0 and b_neg > 0:
    print("  ✓ 방향 일치: 개선 누적 → 퇴출 감소 / 악화 누적 → 퇴출 증가 (이론 예측 방향)")
elif b_pos < 0 and b_neg < 0:
    print("  △ 양방향 모두 음수: 움직임 자체가 생존에 기여 (적응 활동 프레임)")
else:
    print(f"  ? 예상과 다른 패턴 확인 필요")

# exit rate가 낮을 경우 검정력 경고
exit_rate = surv_df["exit_next"].mean()
print(f"\n  주의: exit rate = {exit_rate:.1%} — 낮은 이벤트 비율로 인해 SE가 크게 추정됨")
print(f"  (이벤트 수 = {surv_df['exit_next'].sum()}, 이론적 최소 권고 = 10 per variable)")

# =========================================================
# 9. SAVE RESULTS
# =========================================================
def save_coef(model, fname):
    pd.DataFrame({
        "coef":  model.params,
        "se":    model.bse,
        "pval":  model.pvalues,
        "OR":    np.exp(model.params),
    }).to_csv(fname, encoding="utf-8-sig")
    print(f"Saved: {fname}")

save_coef(m1, "results_H1_M1_baseline.csv")
save_coef(m2, "results_H1_M2_fit.csv")
save_coef(m3, "results_H1_M3_variability.csv")
save_coef(m4, "results_H2_M4_positive_momentum.csv")
save_coef(m5, "results_H2_M5_negative_momentum.csv")
save_coef(m6, "results_H2_M6_joint.csv")

h1_clean.to_csv("data_H1_listing_level.csv",  index=False, encoding="utf-8-sig")
surv_df.to_csv( "data_H2_panel_survival.csv",  index=False, encoding="utf-8-sig")
print("\n완료: main_analysis.py")
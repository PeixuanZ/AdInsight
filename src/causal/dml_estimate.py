# T_has_urgency: 5.4%，too few samples

"""
Week 5: Double Machine Learning (DML) causal estimation
estimate every treatment's CATE to Y_mean_ictr
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from econml.dml import LinearDML, CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_parquet(PROCESSED_DIR / "dataset_causal.parquet")
    with open(PROCESSED_DIR / "causal_meta.json") as f:
        meta = json.load(f)
    return df, meta


def run_linear_dml(df, treatment, outcome, confounders):
    """
    LinearDML: estimate Average Treatment Effect (ATE)
    """

    X = df[confounders].values
    T = df[treatment].values
    Y = df[outcome].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    est = LinearDML(
        model_y=GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            random_state=42
        ),
        model_t=LogisticRegressionCV(
            cv=3,
            max_iter=1000
        ),
        discrete_treatment=True,
        cv=3,
        random_state=42
    )

    est.fit(Y, T, X=X_scaled)

    ate = est.ate(X_scaled)
    ate_inf = est.ate_interval(
        X_scaled,
        alpha=0.05
    )
    se = (float(ate_inf[1]) - float(ate_inf[0])) / (2 * 1.96)
    z = ate / se if se > 0 else 0.0
    p_value = 2 * (1 - norm.cdf(abs(z)))

    # propensity diagnostics
    prop_model = LogisticRegressionCV(
        cv=3,
        max_iter=1000,
        random_state=42
    )

    prop_model.fit(X_scaled, T)

    propensity = (
        prop_model
        .predict_proba(X_scaled)[:, 1]
    )

    return {
        "treatment": treatment,
        "outcome": outcome,
        "ate": float(ate),
        "ci_lower": float(ate_inf[0]),
        "ci_upper": float(ate_inf[1]),
        "p_value": float(p_value),
        "significant_raw":
            float(ate_inf[0]) > 0
            or float(ate_inf[1]) < 0,

        "treated_pct": float(T.mean()),

        "prop_min": float(
            propensity.min()
        ),

        "prop_mean": float(
            propensity.mean()
        ),

        "prop_max": float(
            propensity.max()
        )
    }

def run_naive_ols(df, treatment, outcome, confounders):
    """
    Naive baseline: plain OLS regression of Y on T + confounders,
    with NO cross-fitting and NO orthogonalization. This is exactly
    the kind of "naive" estimate DML is designed to improve on —
    comparing the two shows how much the confounder-adjustment
    machinery actually changes the answer.
    """
    X = df[[treatment] + confounders].values
    y = df[outcome].values
    X_with_const = sm.add_constant(X)

    model = sm.OLS(y, X_with_const).fit()

    # column 0 is the constant, column 1 is the treatment
    naive_ate = model.params[1]
    ci = model.conf_int()[1]
    naive_p = model.pvalues[1]

    return {
        "treatment": treatment,
        "naive_ate": float(naive_ate),
        "naive_ci_lower": float(ci[0]),
        "naive_ci_upper": float(ci[1]),
        "naive_p": float(naive_p)
    }



def run_placebo_test(df, treatment, outcome, confounders, n_shuffles=5, seed=42):
    """
    Refutation check: shuffle the treatment column randomly and rerun
    LinearDML. A real causal effect should vanish once treatment is
    decoupled from outcome — if the "ATE" stays large and significant
    on shuffled data, something in the pipeline is leaking rather than
    estimating a genuine effect.
    """
    rng = np.random.RandomState(seed)
    placebo_ates = []

    for i in range(n_shuffles):
        df_shuffled = df.copy()
        df_shuffled[treatment] = rng.permutation(df_shuffled[treatment].values)
        res = run_linear_dml(df_shuffled, treatment, outcome, confounders)
        placebo_ates.append(res["ate"])
        print(f"    placebo shuffle {i+1}/{n_shuffles}: "
              f"ATE={res['ate']:.6f} p={res['p_value']:.4f}")

    placebo_ates = np.array(placebo_ates)
    return {
        "treatment": treatment,
        "placebo_ate_mean": float(placebo_ates.mean()),
        "placebo_ate_std": float(placebo_ates.std()),
        "n_shuffles": n_shuffles
    }

def run_causal_forest(df, treatment, outcome, confounders):
    """
    CausalForestDML: 估计 CATE (Conditional Average Treatment Effect)
    适合发现哪类广告受 treatment 影响更大
    """
    X = df[confounders].values
    T = df[treatment].values
    Y = df[outcome].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    est = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
        model_t=LogisticRegressionCV(cv=3, max_iter=1000),
        discrete_treatment=True,
        n_estimators=200,
        cv=3,
        random_state=42
    )
    est.fit(Y, T, X=X_scaled)

    # 每个广告的 CATE
    cate = est.effect(X_scaled)
    cate_inf = est.effect_interval(X_scaled, alpha=0.05)

    return est, cate, cate_inf, scaler


def main():
    print("Loading data...")
    df, meta = load_data()

    treatments = meta["treatments"]
    outcome = "Y_mean_ictr"
    confounders = meta["confounders"]

    # 跳过样本太少的 treatment
    skip = ["T_has_urgency", "T_late_cta"]
    treatments_to_run = [t for t in treatments if t not in skip]
    print("\n=== Treatment prevalence ===")

    for t in treatments:
        pct = df[t].mean() * 100

        print(
            f"{t:20s} "
            f"{pct:.1f}% "
            f"({df[t].sum():.0f}/{len(df)})"
        )

    print(f"\nRunning LinearDML for {len(treatments_to_run)} treatments...")
    print(f"Outcome: {outcome}")
    print(f"Confounders: {len(confounders)}")

    # ── Step 1: LinearDML → ATE ──────────────────────────
    results = []
    for t in treatments_to_run:
        print(f"\n  Estimating {t}...")
        res = run_linear_dml(df, t, outcome, confounders)
        results.append(res)
        sig = "✓ SIGNIFICANT (raw p)" if res["significant_raw"] else "✗ not significant"
        print(f"    ATE: {res['ate']:.6f} "
              f"[{res['ci_lower']:.6f}, {res['ci_upper']:.6f}] "
              f"p={res['p_value']:.4f} {sig}")

    df_results = pd.DataFrame(results)

    # ── Naive OLS baseline (no confounder adjustment) ─────────
    # Run alongside DML for every treatment so we can show, side by
    # side, how much the DML adjustment actually changes the estimate.
    print("\n=== Naive OLS baseline (no orthogonalization) ===")
    naive_results = []
    for t in treatments_to_run:
        naive_res = run_naive_ols(df, t, outcome, confounders)
        naive_results.append(naive_res)
        print(f"  {t:20s} naive_ate={naive_res['naive_ate']:.6f} "
              f"p={naive_res['naive_p']:.4f}")

    df_naive = pd.DataFrame(naive_results)
    df_results = df_results.merge(df_naive, on="treatment")
    df_results["ate_diff_dml_minus_naive"] = df_results["ate"] - df_results["naive_ate"]

    df_results.to_csv(OUTPUTS_DIR / "ate_vs_naive_comparison.csv", index=False)
    print("✓ Saved outputs/ate_vs_naive_comparison.csv")

    # ── FDR correction across all treatments tested ──────────
    # We ran multiple independent significance tests (one per
    # treatment). At alpha=0.05, ~5% false positives are expected
    # by chance in each test even if nothing were real — with
    # ~9 treatments that adds up. Benjamini-Hochberg controls the
    # false discovery rate across the whole batch.
    reject, q_values, _, _ = multipletests(
        df_results["p_value"].values, alpha=0.05, method="fdr_bh"
    )
    df_results["q_value"] = q_values
    df_results["significant_fdr"] = reject

    n_raw = df_results["significant_raw"].sum()
    n_fdr = df_results["significant_fdr"].sum()
    print(f"\n=== Multiple-testing correction (Benjamini-Hochberg FDR) ===")
    print(f"Significant at raw p<0.05:      {n_raw}/{len(df_results)}")
    print(f"Significant after FDR (q<0.05): {n_fdr}/{len(df_results)}")
    if n_raw != n_fdr:
        dropped = df_results[
            df_results["significant_raw"] & ~df_results["significant_fdr"]
        ]["treatment"].tolist()
        print(f"Dropped after correction: {dropped}")

    print("\n=== ATE Results ===")
    print(df_results.to_string(index=False))

    print("\n=== DML vs. Naive OLS comparison ===")
    print(df_results[["treatment", "ate", "naive_ate",
                       "ate_diff_dml_minus_naive"]].to_string(index=False))

    # ── Placebo refutation test ──────────────────────────────
    # Run on the top 3 treatments by |ATE| — if the pipeline is sound,
    # shuffled-treatment ATEs should be much smaller than the real ones.
    print("\n=== Placebo refutation test (top 3 by |ATE|) ===")
    top3 = df_results.reindex(
        df_results["ate"].abs().sort_values(ascending=False).index
    ).head(3)

    placebo_results = []
    for _, row in top3.iterrows():
        t = row["treatment"]
        print(f"\n  Placebo test for {t} (real ATE={row['ate']:.6f})...")
        placebo = run_placebo_test(df, t, outcome, confounders)
        placebo["real_ate"] = row["ate"]
        placebo_results.append(placebo)

    pd.DataFrame(placebo_results).to_csv(
        OUTPUTS_DIR / "placebo_results.csv", index=False
    )
    print("\n✓ Saved outputs/placebo_results.csv")

    df_results.to_csv(OUTPUTS_DIR / "ate_results.csv", index=False)
    print(f"\n Saved to outputs/ate_results.csv")


    # ── Step 2: CausalForest → CATE for best treatment ──
    # Prefer a treatment that's still significant AFTER FDR correction —
    # picking by raw |ATE| alone risks building the whole heterogeneity
    # analysis (Step 3+) on an effect that was actually noise.
    fdr_sig = df_results[df_results["significant_fdr"]]
    candidates = fdr_sig if len(fdr_sig) > 0 else df_results
    if len(fdr_sig) == 0:
        print("\n⚠ No treatment survived FDR correction — falling back to "
              "largest |ATE| overall. Treat the CATE analysis below as "
              "exploratory, not confirmatory.")

    best_treatment = (
        candidates.iloc[
            candidates["ate"]
            .abs()
            .argmax()
        ]["treatment"]
    )

    print(
        f"\nUsing treatment with largest "
        f"absolute ATE (FDR-significant preferred): {best_treatment}"
    )

    print(f"\n=== CausalForest CATE for {best_treatment} ===")
    est_cf, cate, cate_inf, scaler = run_causal_forest(
        df, best_treatment, outcome, confounders
    )

    df["cate"] = cate
    df["cate_lower"] = cate_inf[0]
    df["cate_upper"] = cate_inf[1]
    df["recommend_treatment"] = (
    df["cate_lower"] > 0
)

    print(f"CATE stats:")
    print(f"  mean:  {cate.mean():.6f}")
    print(f"  std:   {cate.std():.6f}")
    print(f"  min:   {cate.min():.6f}")
    print(f"  max:   {cate.max():.6f}")
    print(f"  % positive: {(cate > 0).mean()*100:.1f}%")

    # 保存带 CATE 的数据集
    df_out = df[["ad_id", "cate", "cate_lower", "cate_upper",
                 best_treatment, outcome] + confounders]
    df_out.to_parquet(PROCESSED_DIR / "dataset_with_cate.parquet", index=False)
    print(f"\n Saved dataset_with_cate.parquet")

    # Feature importance from causal forest
    print("\n=== Top confounders driving heterogeneity ===")
    importances = est_cf.feature_importances_
    fi = pd.Series(importances, index=confounders).sort_values(ascending=False)
    print(fi.head(8))

    print("\n=== Quartile Segment Analysis ===")

    top_features = fi.head(5).index.tolist()
    with open(
        OUTPUTS_DIR / "top_heterogeneity_features.json",
        "w"
    ) as f:
        json.dump(top_features, f, indent=2)

    quartile_rows = []

    for feature in top_features:

        print(f"\n{feature}")

        temp = df.copy()

        temp["bin"] = pd.qcut(
            temp[feature],
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop"
        )

        quartile_stats = (
            temp.groupby("bin")["cate"]
                .mean()
        )

        print(quartile_stats)

        
        stats = temp.groupby("bin")["cate"].mean()

        for q, val in stats.items():
            quartile_rows.append({
                "feature": feature,
                "quartile": q,
                "cate_mean": val
            })
    
    pd.DataFrame(quartile_rows).to_csv(
    OUTPUTS_DIR / "quartile_analysis.csv",
    index=False
)


    fi.to_csv(OUTPUTS_DIR / "feature_importance.csv")
    print(" Saved feature_importance.csv")
    top_features = (
        fi.head(5)
        .index
        .tolist()
    )

    with open(
        OUTPUTS_DIR /
        "top_heterogeneity_features.json",
        "w"
    ) as f:

        json.dump(
            top_features,
            f,
            indent=2
        )
        

if __name__ == "__main__":
    main()

'''
Input:
    Ad creative features

Methods:
    Double Machine Learning
    Causal Forests

outputs/

    ate_results.csv

    feature_importance.csv

    quartile_analysis.csv

    top_heterogeneity_features.json

data/processed/

dataset_with_cate.parquet
'''
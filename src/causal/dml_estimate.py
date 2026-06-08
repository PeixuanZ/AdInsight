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
            max_depth=3
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

    # propensity diagnostics
    prop_model = LogisticRegressionCV(
        cv=3,
        max_iter=1000
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
        "significant":
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
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3),
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
        sig = "✓ SIGNIFICANT" if res["significant"] else "✗ not significant"
        print(f"    ATE: {res['ate']:.6f} "
              f"[{res['ci_lower']:.6f}, {res['ci_upper']:.6f}] {sig}")

    df_results = pd.DataFrame(results)
    print("\n=== ATE Results ===")
    print(df_results.to_string(index=False))

    df_results.to_csv(OUTPUTS_DIR / "ate_results.csv", index=False)
    print(f"\n Saved to outputs/ate_results.csv")

    # ── Step 2: CausalForest → CATE for best treatment ──
    # 选 ATE 最大且显著的 treatment 做 CATE
    best_treatment = (
        df_results.iloc[
            df_results["ate"]
            .abs()
            .argmax()
        ]["treatment"]
    )

    print(
        f"\nUsing treatment with largest "
        f"absolute ATE: {best_treatment}"
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
"""
Stage 2 — Module 1: Adaptive Multimodal Router
cheap transcript + ICTR feature predication whether needs CLIP or not
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings("ignore")

ROOT_DIR      = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR   = ROOT_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


# ── Step 1: 定义 routing target ───────────────────────────────
def build_routing_target_v2(df: pd.DataFrame) -> pd.Series:
    """
    严谨的 routing target：
    用 text-only 特征跑 5-fold out-of-fold 预测，
    预测误差大的广告 = 文本特征不够用 = 需要 CLIP
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import KFold

    # 只用便宜特征预测 outcome
    text_features = [
        "n_promo_kw", "n_cta_kw", "n_price_kw", "n_urgency_kw",
        "speech_rate", "n_segments", "segment_density",
        "first_promo_time", "first_cta_time",
        "duration_sec", "transcript_duration",
        "T_promo_first_5s", "T_cta_early", "T_cta_late"
    ]
    available = [f for f in text_features if f in df.columns]

    X_text = df[available].fillna(df[available].median()).values
    y      = df["mean_ictr"].values

    # 5-fold OOF 预测
    oof_preds = np.zeros(len(df))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(X_text):
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=42
        )
        model.fit(X_text[train_idx], y[train_idx])
        oof_preds[val_idx] = model.predict(X_text[val_idx])

    # 预测残差（绝对误差）
    residuals = np.abs(y - oof_preds)

    # 残差大 = 文本无法解释 = 需要 CLIP
    threshold  = np.median(residuals)
    needs_clip = (residuals > threshold).astype(int)

    print(f"Text-only OOF R²: "
          f"{1 - residuals.var() / np.var(y):.4f}")
    print(f"Mean residual: {residuals.mean():.5f}")
    print(f"Needs CLIP: {needs_clip.sum()} ({needs_clip.mean()*100:.1f}%)")

    return pd.Series(needs_clip, index=df.index), residuals


# ── Step 2: cheap features（不需要 CLIP 就能计算）────────────────────
CHEAP_FEATURES = [
    # ICTR shape features
    "mean_ictr", "max_ictr", "ictr_std", "ictr_slope",
    "peak_relative_pos", "late_early_ratio",
    "early_mean_ictr", "late_mean_ictr",
    # Transcript features
    "n_segments", "speech_rate", "segment_density",
    "n_price_kw", "n_promo_kw", "n_urgency_kw",
    "n_product_kw", "n_cta_kw",
    "has_price", "has_promo", "has_urgency", "has_cta",
    "first_promo_time", "first_cta_time",
    # Structural
    "duration_sec", "n_frames", "transcript_duration",
]


def get_cheap_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only cheap (non-CLIP) features."""
    available = [f for f in CHEAP_FEATURES if f in df.columns]
    missing   = [f for f in CHEAP_FEATURES if f not in df.columns]
    if missing:
        print(f"  Missing features (will skip): {missing}")
    X = df[available].copy()
    # Fill -1 sentinel values with median
    X = X.replace(-1, np.nan)
    X = X.fillna(X.median())
    return X


# ── Step 3: training adaptive router ─────────────────────────────────────────
def train_router(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Train and compare multiple routing classifiers.
    Returns the best model + cross-validation results.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "logistic":  LogisticRegression(max_iter=1000, C=1.0),
        "gbm":       GradientBoostingClassifier(
                         n_estimators=100, max_depth=3, random_state=42),
        "rf":        RandomForestClassifier(
                         n_estimators=100, max_depth=5, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("\nCross-validation AUC (5-fold):")
    for name, model in models.items():
        scores = cross_val_score(
            model, X_scaled, y, cv=cv, scoring="roc_auc"
        )
        results[name] = {
            "auc_mean": scores.mean(),
            "auc_std":  scores.std(),
            "model":    model
        }
        print(f"  {name:12s}  AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    # Best model
    best_name = max(results, key=lambda k: results[k]["auc_mean"])
    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name} "
          f"(AUC = {results[best_name]['auc_mean']:.4f})")

    # Calibrate for reliable probability estimates
    calibrated = CalibratedClassifierCV(best_model, cv=5, method="isotonic")
    calibrated.fit(X_scaled, y)

    return {
        "model":      calibrated,
        "scaler":     scaler,
        "best_name":  best_name,
        "cv_results": {k: {"auc_mean": v["auc_mean"],
                           "auc_std":  v["auc_std"]}
                       for k, v in results.items()},
        "features":   list(X.columns)
    }


# ── Step 4: Budget-aware routing ──────────────────────────────
def apply_budget_routing(
    df: pd.DataFrame,
    router: dict,
    clip_budget: float = 0.30
) -> pd.DataFrame:
    """
    Given a CLIP budget (e.g. 0.30 = only 30% of ads get CLIP),
    route each ad to cheap or expensive path.

    Strategy: sort by routing probability descending,
    assign CLIP to top-k ads where k = budget * N.
    """
    X      = get_cheap_features(df)
    scaler = router["scaler"]
    model  = router["model"]

    X_scaled = scaler.transform(X)
    proba    = model.predict_proba(X_scaled)[:, 1]  # P(needs_clip)

    df = df.copy()
    df["clip_proba"]   = proba
    df["clip_rank"]    = pd.Series(proba).rank(ascending=False).values

    n_clip = int(len(df) * clip_budget)
    df["routed_to_clip"] = (df["clip_rank"] <= n_clip).astype(int)

    print(f"\nBudget = {clip_budget*100:.0f}% → "
          f"{n_clip} ads routed to CLIP, "
          f"{len(df)-n_clip} to text-only")

    return df


# ── Step 5: 评估路由质量 ───────────────────────────────────────
def evaluate_routing(df: pd.DataFrame, y_true: pd.Series) -> dict:
    """
    Compare attribution quality: full CLIP vs routed (budget-constrained).
    Metric: how many truly-needs-CLIP ads did we correctly route to CLIP?
    """
    results = {}
    for budget in [0.10, 0.20, 0.30, 0.40, 0.50]:
        routed = (df["clip_rank"] <= int(len(df) * budget)).astype(int)

        # Recall of needs-clip ads
        true_pos  = ((routed == 1) & (y_true == 1)).sum()
        all_pos   = y_true.sum()
        recall    = true_pos / all_pos if all_pos > 0 else 0

        # Precision
        pred_pos  = routed.sum()
        precision = true_pos / pred_pos if pred_pos > 0 else 0

        results[budget] = {
            "budget_pct":   float(budget * 100),
            "n_clip":       int(pred_pos),
            "recall":       round(float(recall), 4),
            "precision":    round(float(precision), 4),
            "f1":           round(float(2 * precision * recall /
                                  (precision + recall + 1e-9)), 4)
        }
        print(f"  Budget {budget*100:.0f}%: "
              f"recall={recall:.3f}  precision={precision:.3f}  "
              f"F1={results[budget]['f1']:.3f}")

    return results

def compare_attribution_quality(
    df: pd.DataFrame,
    residuals: np.ndarray
):
    """
    验证：被路由到 CLIP 的广告，text-only 残差是否确实更大？
    """
    df = df.copy()
    df["text_residual"] = residuals

    routed_clip    = df[df["routed_to_clip"] == 1]["text_residual"]
    routed_text    = df[df["routed_to_clip"] == 0]["text_residual"]

    from scipy import stats
    t_stat, p_val = stats.ttest_ind(routed_clip, routed_text)

    print(f"  Routed to CLIP   — mean residual: {routed_clip.mean():.5f}")
    print(f"  Routed text-only — mean residual: {routed_text.mean():.5f}")
    print(f"  t-test p-value: {p_val:.4f} "
          f"{'✓ significant' if p_val < 0.05 else '✗ not significant'}")
    print(f"  Lift in residual: "
          f"{(routed_clip.mean()-routed_text.mean())/routed_text.mean()*100:+.1f}%")

# ── Main ──────────────────────────────────────────────────────
def main():
    print("Loading data...")
    df = pd.read_parquet(PROCESSED_DIR / "dataset_causal.parquet")
    df["ad_id"] = df["ad_id"].astype(str)

    print(f"Dataset: {len(df)} ads, {len(df.columns)} features")

    # Step 1: routing target
    print("\n=== Step 1: Build routing target (text-only residuals) ===")
    y, residuals = build_routing_target_v2(df)
    df["needs_clip"] = y
    df["text_residual"] = residuals

    # Step 2: cheap features
    print("\n=== Step 2: Extract cheap features ===")
    X = get_cheap_features(df)
    print(f"Using {len(X.columns)} cheap features")

    # Step 3: train router: predict which ads, text-only larger variance
    print("\n=== Step 3: Train routing classifier ===")
    router = train_router(X, y)

    # Step 4: apply budget routing
    print("\n=== Step 4: Budget-aware routing ===")
    df = apply_budget_routing(df, router, clip_budget=0.30)

    # Step 5: evaluate
    print("\n=== Step 5: Evaluate routing quality ===")
    eval_results = evaluate_routing(df, y)

    # Step 6: validate routing improves attribution quality
    print("\n=== Step 6: Attribution quality comparison ===")
    compare_attribution_quality(df, residuals)

    # Feature importance
    print("\n=== Top routing features ===")
    best_name = router["best_name"]
    if best_name in ["gbm", "rf"]:
        base_model = router["model"].calibrated_classifiers_[0].estimator
        fi = pd.Series(
            base_model.feature_importances_,
            index=router["features"]
        ).sort_values(ascending=False)
        print(fi.head(10))
        fi.to_csv(OUTPUTS_DIR / "router_feature_importance.csv")

    # Save outputs
    df[["ad_id", "needs_clip", "clip_proba",
        "clip_rank", "routed_to_clip"]].to_parquet(
        PROCESSED_DIR / "routing_decisions.parquet", index=False
    )

    with open(OUTPUTS_DIR / "routing_eval.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    router_meta = {
        "best_model":  router["best_name"],
        "cv_results":  router["cv_results"],
        "features":    router["features"],
        "n_features":  len(router["features"])
    }
    with open(OUTPUTS_DIR / "router_meta.json", "w") as f:
        json.dump(router_meta, f, indent=2)

    print("\n✓ Saved routing_decisions.parquet")
    print(" Saved routing_eval.json")
    print(" Saved router_meta.json")

    # Summary
    print("\n=== Summary ===")
    print(f"Best router AUC: "
          f"{router['cv_results'][router['best_name']]['auc_mean']:.4f}")
    print(f"At 30% budget: "
          f"recall={eval_results[0.30]['recall']:.3f}, "
          f"F1={eval_results[0.30]['f1']:.3f}")


if __name__ == "__main__":
    main()

    
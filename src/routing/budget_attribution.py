"""
Stage 2 — Module 2: Budget-Aware Attribution
在不同计算预算下，对比归因质量的 accuracy-cost tradeoff
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

ROOT_DIR      = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR   = ROOT_DIR / "outputs"


def load_data():
    df         = pd.read_parquet(PROCESSED_DIR / "dataset_causal.parquet")
    df_routing = pd.read_parquet(PROCESSED_DIR / "routing_decisions.parquet")
    df_cate    = pd.read_parquet(PROCESSED_DIR / "dataset_with_cate.parquet")
    df["ad_id"]         = df["ad_id"].astype(str)
    df_routing["ad_id"] = df_routing["ad_id"].astype(str)
    df_cate["ad_id"]    = df_cate["ad_id"].astype(str)

    df = df.merge(df_routing, on="ad_id", how="left")
    df = df.merge(df_cate[["ad_id", "cate"]], on="ad_id", how="left")
    return df


def get_feature_sets(df: pd.DataFrame) -> dict:
    """
    定义三种特征集合，代表不同计算成本。
    """
    text_features = [
        "n_promo_kw", "n_cta_kw", "n_price_kw", "n_urgency_kw",
        "speech_rate", "n_segments", "segment_density",
        "first_promo_time", "first_cta_time",
        "duration_sec", "transcript_duration",
        "T_promo_first_5s", "T_cta_early", "T_cta_late",
        "ictr_std", "ictr_slope", "late_early_ratio",
        "early_mean_ictr", "late_mean_ictr",
    ]
    vis_features = [c for c in df.columns if c.startswith("vis_")
                    and not c.startswith("vis_z_")]
    text_available = [f for f in text_features if f in df.columns]
    vis_available  = [f for f in vis_features  if f in df.columns]

    return {
        "text_only": text_available,
        "text_plus_clip": text_available + vis_available,
        "routed_30pct": None,  # handled separately
    }


def evaluate_attribution(
    X: np.ndarray,
    y: np.ndarray,
    label: str,
    n_folds: int = 5
) -> dict:
    """
    5-fold OOF evaluation of attribution quality.
    Metric: R^2 and RMSE of mean_ictr prediction.
    """
    oof = np.zeros(len(y))
    kf  = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(X):
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=42
        )
        model.fit(X[train_idx], y[train_idx])
        oof[val_idx] = model.predict(X[val_idx])

    r2   = r2_score(y, oof)
    rmse = np.sqrt(mean_squared_error(y, oof))

    print(f"  {label:30s}  R^2={r2:.4f}  RMSE={rmse:.5f}")
    return {
        "label":    label,
        "r2":       round(float(r2), 4),
        "rmse":     round(float(rmse), 5),
        "oof_preds": oof
    }


def evaluate_routed_attribution(
    df: pd.DataFrame,
    text_features: list,
    vis_features: list,
    clip_budget: float = 0.30,
    outcome = 'cate'
) -> dict:
    """
    Routed attribution：
    - CLIP 组（top clip_budget%）用 text + visual features
    - Text 组用 text-only features
    合并 OOF 预测，评估整体质量。
    """
    y = df["cate"].values if "cate" in df.columns else df["mean_ictr"].values


    # 按 routing 决策分组
    clip_mask = df["routed_to_clip"].values == 1
    text_mask = ~clip_mask

    oof = np.zeros(len(df))
    kf  = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(df):
        # CLIP 组
        clip_train = train_idx[clip_mask[train_idx]]
        clip_val   = val_idx[clip_mask[val_idx]]
        if len(clip_val) > 0:
            X_clip = df[text_features + vis_features].fillna(0).values
            m = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, random_state=42
            )
            m.fit(X_clip[clip_train], y[clip_train])
            oof[clip_val] = m.predict(X_clip[clip_val])

        # Text 组
        text_train = train_idx[text_mask[train_idx]]
        text_val   = val_idx[text_mask[val_idx]]
        if len(text_val) > 0:
            X_text = df[text_features].fillna(0).values
            m2 = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, random_state=42
            )
            m2.fit(X_text[text_train], y[text_train])
            oof[text_val] = m2.predict(X_text[text_val])

    r2   = r2_score(y, oof)
    rmse = np.sqrt(mean_squared_error(y, oof))
    label = f"routed ({int(clip_budget*100)}% CLIP)"

    print(f"  {label:30s}  R^2={r2:.4f}  RMSE={rmse:.5f}")
    return {
        "label":     label,
        "r2":        round(float(r2), 4),
        "rmse":      round(float(rmse), 5),
        "oof_preds": oof,
        "clip_budget": clip_budget
    }


def compute_cost_accuracy_curve(
    df: pd.DataFrame,
    text_features: list,
    vis_features: list,
    outcome: str = 'cate'
) -> list:
    """
    Sweep clip_budget from 0% to 100%，
    计算每个 budget 下的 R^2。
    这是 budget-aware attribution 的核心曲线。
    """
    y       = df[outcome].values
    budgets = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30,
               0.40, 0.50, 0.60, 0.70, 0.80, 1.0]
    curve   = []

    print("\nSweeping budgets:")
    for budget in budgets:
        if budget == 0.0:
            # Pure text-only
            X    = df[text_features].fillna(0).values
            res  = evaluate_attribution(X, y, f"text-only (0%)")
        elif budget == 1.0:
            # Full CLIP
            X    = df[text_features + vis_features].fillna(0).values
            res  = evaluate_attribution(X, y, f"full CLIP (100%)")
        else:
            # Routed
            n_clip = int(len(df) * budget)
            # Assign top-n by clip_proba to CLIP group
            df_tmp = df.copy()
            df_tmp["routed_to_clip"] = (
                df_tmp["clip_proba"].rank(ascending=False) <= n_clip
            ).astype(int)
            res = evaluate_routed_attribution(
                df_tmp, text_features, vis_features, budget
            )

        curve.append({
            "budget":     float(budget),
            "budget_pct": float(budget * 100),
            "r2":         res["r2"],
            "rmse":       res["rmse"],
            "n_clip":     int(len(df) * budget)
        })

    return curve


def plot_budget_curves(curve: list, results: dict):
    """Generate budget-accuracy tradeoff plots."""
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    budgets = [c["budget_pct"] for c in curve]
    r2s     = [c["r2"]         for c in curve]
    rmses   = [c["rmse"]       for c in curve]
    n_clips = [c["n_clip"]     for c in curve]

    # ── Plot 1: R² vs budget ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(budgets, r2s, color="#1D9E75", linewidth=2.5,
             marker="o", markersize=5)
    ax1.axhline(r2s[0],  color="#888780", linestyle="--",
                linewidth=1, label=f"Text-only R²={r2s[0]:.3f}")
    ax1.axhline(r2s[-1], color="#7F77DD", linestyle="--",
                linewidth=1, label=f"Full CLIP R²={r2s[-1]:.3f}")
    ax1.set_xlabel("CLIP budget (%)")
    ax1.set_ylabel("Attribution R²")
    ax1.set_title("Accuracy vs CLIP budget")
    ax1.legend(fontsize=8)

    # ── Plot 2: RMSE vs budget ───────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(budgets, rmses, color="#D85A30", linewidth=2.5,
             marker="o", markersize=5)
    ax2.set_xlabel("CLIP budget (%)")
    ax2.set_ylabel("RMSE")
    ax2.set_title("Error vs CLIP budget")

    # ── Plot 3: marginal R² gain per % of budget ─────────────
    ax3 = fig.add_subplot(gs[1, 0])
    marginal = np.diff(r2s) / np.diff(budgets)
    ax3.bar(budgets[1:], marginal, width=3,
            color="#7F77DD", alpha=0.8, edgecolor="white")
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax3.set_xlabel("CLIP budget (%)")
    ax3.set_ylabel("ΔR² per 1% budget")
    ax3.set_title("Marginal return on CLIP investment")

    # ── Plot 4: Efficiency frontier ──────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    r2_gain  = [r - r2s[0] for r in r2s]
    ax4.plot(n_clips, r2_gain, color="#BA7517", linewidth=2.5,
             marker="o", markersize=5)
    ax4.set_xlabel("Number of ads processed with CLIP")
    ax4.set_ylabel("R² gain over text-only")
    ax4.set_title("Efficiency frontier")

    # Mark sweet spot (elbow)
    gains    = np.array(r2_gain)
    budgets_arr = np.array(budgets)
    if len(gains) > 2:
        # Elbow = max marginal gain dropoff
        marginal2 = np.diff(gains) / (np.diff(budgets_arr) + 1e-9)
        elbow_idx = np.argmax(marginal2) + 1
        ax4.axvline(n_clips[elbow_idx], color="#D85A30",
                    linestyle="--", linewidth=1.5,
                    label=f"Sweet spot: {budgets[elbow_idx]:.0f}% budget")
        ax4.legend(fontsize=8)

    plt.suptitle("Budget-Aware Attribution — AdLens Stage 2",
                 fontsize=13, y=1.01)
    out = OUTPUTS_DIR / "budget_attribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved {out}")
    plt.show()


def main():
    print("Loading data...")
    df = load_data()

    feature_sets = get_feature_sets(df)
    text_features = feature_sets["text_only"]
    vis_features  = [c for c in df.columns if c.startswith("vis_")
                     and not c.startswith("vis_z_")]
    vis_features  = [f for f in vis_features if f in df.columns]

    # y = df["mean_ictr"].values
    y = df["cate"].dropna().values
    df = df.dropna(subset=["cate"])
    print(f"Outcome: cate  (n={len(df)}, mean={y.mean():.5f}, std={y.std():.5f})")

    # ── Baseline comparisons ──────────────────────────────────
    print("\n=== Baseline attribution quality ===")
    res_text = evaluate_attribution(
        df[text_features].fillna(0).values, y, "text-only"
    )
    res_full = evaluate_attribution(
        df[text_features + vis_features].fillna(0).values, y, "full CLIP"
    )
    res_routed = evaluate_routed_attribution(
        df, text_features, vis_features, clip_budget=0.30
    )

    print(f"\n  R² gain (text→full CLIP):   "
          f"{res_full['r2'] - res_text['r2']:+.4f}")
    print(f"  R² gain (text→routed 30%):  "
          f"{res_routed['r2'] - res_text['r2']:+.4f}")
    print(f"  Efficiency (routed/full):   "
          f"{(res_routed['r2']-res_text['r2'])/(res_full['r2']-res_text['r2']+1e-9)*100:.1f}%"
          f" of full-CLIP gain at 30% cost")

    # ── Budget sweep ──────────────────────────────────────────
    print("\n=== Budget sweep ===")
    curve = compute_cost_accuracy_curve(df, text_features, vis_features,outcome = 'cate')

    # ── Save + plot ───────────────────────────────────────────
    with open(OUTPUTS_DIR / "budget_curve.json", "w") as f:
        json.dump(curve, f, indent=2)
    print("✓ Saved budget_curve.json")

    plot_budget_curves(curve, {})

    # Summary table
    print("\n=== Budget summary table ===")
    df_curve = pd.DataFrame(curve)
    df_curve["r2_gain"]    = df_curve["r2"] - df_curve["r2"].iloc[0]
    df_curve["efficiency"] = (df_curve["r2_gain"] /
                               (df_curve["r2"].iloc[-1] -
                                df_curve["r2"].iloc[0] + 1e-9) * 100)
    print(df_curve[["budget_pct", "n_clip", "r2",
                     "r2_gain", "efficiency"]].to_string(index=False))
    df_curve.to_csv(OUTPUTS_DIR / "budget_summary.csv", index=False)
    print(" Saved budget_summary.csv")


if __name__ == "__main__":
    main()

'''
Key observation:
1. mean ictr: 在 AdsTrace 数据集上，ICTR 统计特征（mean_ictr, ictr_std）已经解释了 99.7% 的方差，视觉特征没有增量信息
'''
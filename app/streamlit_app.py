"""
Week 8: AdInsight Streamlit Demo
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR      = Path(__file__).parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR   = ROOT_DIR / "outputs"

st.set_page_config(
    page_title="AdInsight",
    page_icon="🎯",
    layout="wide"
)

# ── Data loading ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df_master = pd.read_parquet(PROCESSED_DIR / "dataset_causal.parquet")
    df_peaks  = pd.read_parquet(PROCESSED_DIR / "conversion_peaks.parquet")
    df_master["ad_id"] = df_master["ad_id"].astype(str)
    df_peaks["ad_id"]  = df_peaks["ad_id"].astype(str)

    # 确认 peaks 里有哪些列
    peak_cols = ["ad_id", "peak_sec", "peak_relative_pos",
                 "trigger_type", "trigger_text", "n_peaks",
                 "ictr_at_peak_ratio"]
    # 只取存在的列
    peak_cols = [c for c in peak_cols if c in df_peaks.columns]

    df = df_master.merge(df_peaks[peak_cols], on="ad_id", how="left")

    # 确保 trigger_type 存在
    if "trigger_type" not in df.columns:
        df["trigger_type"] = "unknown"
    if "peak_relative_pos" not in df.columns:
        df["peak_relative_pos"] = np.nan
    if "peak_sec" not in df.columns:
        df["peak_sec"] = np.nan

    with open(OUTPUTS_DIR / "curve_comparisons.json") as f:
        curve_results = json.load(f)

    portfolio_insight = ""
    insight_path = OUTPUTS_DIR / "portfolio_insight.txt"
    if insight_path.exists():
        portfolio_insight = insight_path.read_text()

    ad_explanations = pd.DataFrame()
    exp_path = OUTPUTS_DIR / "ad_explanations.csv"
    if exp_path.exists():
        ad_explanations = pd.read_csv(exp_path)
        ad_explanations["ad_id"] = ad_explanations["ad_id"].astype(str)

    return df, curve_results, portfolio_insight, ad_explanations
    
df, curve_results, portfolio_insight, ad_explanations = load_data()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("AdInsight")
st.sidebar.caption("Multimodal causal attribution for video ads")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Temporal Triggers", "Ad Explorer",
     "Causal Insights", "Stage 2: Adaptive Routing"]
)

# ══════════════════════════════════════════════════════════════
# Page 1: Overview
# ══════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("AdInsight — Ad Attribution Dashboard")
    st.markdown("**Causal-inspired attribution framework for e-commerce video ads**")

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Ads", f"{len(df):,}")
    col2.metric("Mean ICTR", f"{df['mean_ictr'].mean():.4f}")
    col3.metric("Peak in Last Quarter",
                f"{df['T_peak_in_last_quarter'].mean()*100:.1f}%")
    col4.metric("Avg Duration", f"{df['duration_sec'].mean():.1f}s")

    st.divider()

    # Portfolio insight
    st.subheader("Portfolio Insight")
    if portfolio_insight:
        st.info(portfolio_insight)
    else:
        st.info("Run Week 7 LLM layer to generate insights.")

    st.divider()

    # ICTR distribution
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("ICTR Distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(df["mean_ictr"], bins=50,
                color="#1D9E75", alpha=0.8, edgecolor="white")
        ax.set_xlabel("Mean ICTR")
        ax.set_ylabel("Count")
        ax.set_title("Conversion rate distribution")
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.subheader("Peak Position Distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(df["peak_relative_pos"].dropna() * 100,
                bins=20, color="#7F77DD", alpha=0.8, edgecolor="white")
        ax.axvline(75, color="#D85A30", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Peak position (% of ad)")
        ax.set_ylabel("Count")
        ax.set_title("When does conversion peak occur?")
        st.pyplot(fig)
        plt.close()

    # Trigger type breakdown
    st.subheader("Conversion Trigger Types")
    ttype = df["trigger_type"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#888780", "#1D9E75", "#7F77DD", "#D85A30", "#BA7517"]
    ttype.plot(kind="bar", ax=ax,
               color=colors[:len(ttype)], edgecolor="white")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.set_title("Content type before conversion peak")
    plt.xticks(rotation=0)
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════
# Page 2: Temporal Triggers
# ══════════════════════════════════════════════════════════════
elif page == "Temporal Triggers":
    st.title("Temporal Trigger Discovery")
    st.markdown("Comparing ICTR curves for treated vs control groups across time.")

    treatment_labels = {
        "T_promo_first_5s":      "Promo in first 5s",
        "T_cta_early":           "CTA in first half",
        "T_cta_late":            "CTA in second half",
        "T_peak_in_last_quarter":"Peak in last quarter"
    }

    selected = st.selectbox(
        "Select treatment",
        list(treatment_labels.keys()),
        format_func=lambda x: treatment_labels[x]
    )

    res = next((r for r in curve_results
                if r["treatment"] == selected), None)

    if res:
        mean_t = np.array(res["mean_treated"])
        mean_c = np.array(res["mean_control"])
        x = np.linspace(0, 100, len(mean_t))

        # Compute CI from aligned data
        df_aligned_path = PROCESSED_DIR / "ictr_aligned.parquet"
        if df_aligned_path.exists():
            df_aligned = pd.read_parquet(df_aligned_path)
            time_cols  = [c for c in df_aligned.columns if c.startswith("t")]
            df_m = df_aligned.merge(df[["ad_id", selected]], on="ad_id")
            treated_v = df_m[df_m[selected] == 1][time_cols].values
            control_v = df_m[df_m[selected] == 0][time_cols].values

            ci_t = 1.96 * treated_v.std(axis=0) / np.sqrt(len(treated_v))
            ci_c = 1.96 * control_v.std(axis=0) / np.sqrt(len(control_v))
        else:
            ci_t = ci_c = np.zeros(len(mean_t))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, mean_t, color="#1D9E75", linewidth=2,
                label=f"Treated (n={res['n_treated']})")
        ax.fill_between(x, mean_t - ci_t, mean_t + ci_t,
                        color="#1D9E75", alpha=0.15)
        ax.plot(x, mean_c, color="#888780", linewidth=2,
                linestyle="--",
                label=f"Control (n={res['n_control']})")
        ax.fill_between(x, mean_c - ci_c, mean_c + ci_c,
                        color="#888780", alpha=0.15)
        for w in res["sig_windows"]:
            ax.axvspan(x[w] - 2.5, x[w] + 2.5,
                       alpha=0.12, color="#D85A30")
        ax.set_xlabel("Ad progress (%)")
        ax.set_ylabel("Mean ICTR")
        ax.set_title(f"{treatment_labels[selected]} — ICTR curve comparison")
        ax.legend()
        st.pyplot(fig)
        plt.close()

        # Stats
        lift = (mean_t.mean() - mean_c.mean()) / (mean_c.mean() + 1e-9) * 100
        col1, col2, col3 = st.columns(3)
        col1.metric("Treated mean ICTR", f"{mean_t.mean():.5f}")
        col2.metric("Control mean ICTR", f"{mean_c.mean():.5f}")
        col3.metric("Lift", f"{lift:+.1f}%",
                    delta_color="normal" if lift > 0 else "inverse")

        st.caption(f"Red shading = statistically significant time windows (p < 0.05). "
                   f"Total significant windows: {len(res['sig_windows'])}")


# ══════════════════════════════════════════════════════════════
# Page 3: Ad Explorer
# ══════════════════════════════════════════════════════════════
elif page == "Ad Explorer":
    st.title("Ad Explorer")
    st.markdown("Browse individual ads and their conversion profiles.")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_ictr = st.slider("Min mean ICTR", 0.0,
                             float(df["mean_ictr"].max()), 0.0,
                             step=0.001, format="%.3f")
    with col2:
        trigger_filter = st.multiselect(
            "Trigger type",
            df["trigger_type"].dropna().unique().tolist(),
            default=df["trigger_type"].dropna().unique().tolist()
        )
    with col3:
        peak_filter = st.selectbox(
            "Peak position",
            ["All", "Early (0-50%)", "Late (50-100%)"]
        )

    df_filtered = df[df["mean_ictr"] >= min_ictr]
    if trigger_filter:
        df_filtered = df_filtered[
            df_filtered["trigger_type"].isin(trigger_filter)
        ]
    if peak_filter == "Early (0-50%)":
        df_filtered = df_filtered[
            df_filtered["peak_relative_pos"] < 0.5
        ]
    elif peak_filter == "Late (50-100%)":
        df_filtered = df_filtered[
            df_filtered["peak_relative_pos"] >= 0.5
        ]

    st.caption(f"Showing {len(df_filtered):,} of {len(df):,} ads")

    # Table
    display_cols = ["ad_id", "mean_ictr", "max_ictr", "duration_sec",
                    "peak_relative_pos", "trigger_type", "n_segments"]
    st.dataframe(
        df_filtered[display_cols]
        .sort_values("mean_ictr", ascending=False)
        .head(100)
        .reset_index(drop=True),
        use_container_width=True
    )

    # Individual ad detail
    st.divider()
    st.subheader("Ad Detail")
    ad_id_input = st.text_input("Enter Ad ID to inspect:")

    if ad_id_input and ad_id_input in df["ad_id"].values:
        row = df[df["ad_id"] == ad_id_input].iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Ad ID:** {ad_id_input}")
            st.markdown(f"**Duration:** {row['duration_sec']:.0f}s")
            st.markdown(f"**Mean ICTR:** {row['mean_ictr']:.5f}")
            st.markdown(f"**Peak position:** "
                        f"{row['peak_relative_pos']*100:.0f}%")
            st.markdown(f"**Trigger type:** {row['trigger_type']}")

        with col2:
            st.markdown("**Transcript:**")
            st.text(str(row.get("full_text", ""))[:400])

        # ICTR time series
        ictr_path = (ROOT_DIR / "data" / "raw" /
                     "ictr" / "ictr" / f"{ad_id_input}.csv")
        if ictr_path.exists():
            df_ictr = pd.read_csv(ictr_path)
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(df_ictr["sec"], df_ictr["ictr"],
                    color="#1D9E75", linewidth=2)
            ax.axvline(row["peak_sec"], color="#D85A30",
                       linestyle="--", linewidth=1.5,
                       label=f"Peak @ {row['peak_sec']:.0f}s")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("ICTR")
            ax.set_title(f"Conversion rate over time — Ad {ad_id_input}")
            ax.legend()
            st.pyplot(fig)
            plt.close()

        # LLM explanation if available
        if len(ad_explanations) > 0:
            exp_row = ad_explanations[
                ad_explanations["ad_id"] == ad_id_input
            ]
            if len(exp_row) > 0:
                st.subheader("LLM Analysis")
                st.success(exp_row["explanation"].iloc[0])

    elif ad_id_input:
        st.warning(f"Ad ID {ad_id_input} not found.")


# ══════════════════════════════════════════════════════════════
# Page 4: Causal Insights
# ══════════════════════════════════════════════════════════════
elif page == "Causal Insights":
    st.title("Causal Insights")

    # ATE results
    ate_path = OUTPUTS_DIR / "ate_results.csv"
    if ate_path.exists():
        st.subheader("Average Treatment Effects (DML)")
        df_ate = pd.read_csv(ate_path)

        fig, ax = plt.subplots(figsize=(8, 4))
        y_pos = range(len(df_ate))
        colors = ["#1D9E75" if sig else "#888780"
                  for sig in df_ate["significant"]]
        ax.barh(y_pos, df_ate["ate"], color=colors, alpha=0.8)
        ax.errorbar(
            df_ate["ate"], y_pos,
            xerr=[df_ate["ate"] - df_ate["ci_lower"],
                  df_ate["ci_upper"] - df_ate["ate"]],
            fmt="none", color="black", capsize=4
        )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(
            df_ate["treatment"].str.replace("T_", "").str.replace("_", " ")
        )
        ax.set_xlabel("ATE on mean ICTR")
        ax.set_title("Causal effect of each treatment (green = significant)")
        st.pyplot(fig)
        plt.close()

        st.dataframe(df_ate, use_container_width=True)

    # Feature importance
    fi_path = OUTPUTS_DIR / "feature_importance.csv"
    if fi_path.exists():
        st.subheader("Heterogeneity Drivers (Causal Forest)")
        df_fi = pd.read_csv(fi_path, header=None,
                            names=["feature", "importance"])
        df_fi = df_fi.sort_values("importance", ascending=True).tail(10)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(df_fi["feature"], df_fi["importance"],
                color="#7F77DD", alpha=0.8)
        ax.set_xlabel("Feature importance")
        ax.set_title("Top drivers of treatment effect heterogeneity")
        st.pyplot(fig)
        plt.close()

    # Quartile analysis
    q_path = OUTPUTS_DIR / "quartile_analysis.csv"
    if q_path.exists():
        st.subheader("CATE by Feature Quartile")
        df_q = pd.read_csv(q_path)
        features = df_q["feature"].unique()
        selected_feat = st.selectbox("Select feature", features)
        df_q_sel = df_q[df_q["feature"] == selected_feat]

        fig, ax = plt.subplots(figsize=(6, 3))
        colors_q = ["#D85A30" if v < 0 else "#1D9E75"
                    for v in df_q_sel["cate_mean"]]
        ax.bar(df_q_sel["quartile"], df_q_sel["cate_mean"],
               color=colors_q, alpha=0.8, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Quartile")
        ax.set_ylabel("Mean CATE")
        ax.set_title(f"Treatment effect by {selected_feat} quartile")
        st.pyplot(fig)
        plt.close()
elif page == "Stage 2: Adaptive Routing":
    st.title("Stage 2 — Adaptive Multimodal Routing")
    st.markdown("Budget-aware attribution: how much CLIP do we actually need?")

    # ── Key metrics ──────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Router AUC", "0.933")
    col2.metric("10% budget precision", "1.000")
    col3.metric("Text-only CATE R²", "0.317")
    col4.metric("Full CLIP CATE R²", "0.792")

    st.divider()

    # ── Budget curve ─────────────────────────────────────────
    budget_path = OUTPUTS_DIR / "budget_curve.json"
    if budget_path.exists():
        with open(budget_path) as f:
            curve = json.load(f)

        df_curve = pd.DataFrame(curve)
        df_curve["r2_gain"] = df_curve["r2"] - df_curve["r2"].iloc[0]
        df_curve["efficiency"] = (
            df_curve["r2_gain"] /
            (df_curve["r2"].iloc[-1] - df_curve["r2"].iloc[0] + 1e-9) * 100
        ).round(1)

        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("CATE R² vs CLIP budget")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(df_curve["budget_pct"], df_curve["r2"],
                    color="#1D9E75", linewidth=2.5, marker="o", markersize=4)
            ax.axhline(df_curve["r2"].iloc[0], color="#888780",
                       linestyle="--", linewidth=1,
                       label=f"Text-only R²={df_curve['r2'].iloc[0]:.3f}")
            ax.axhline(df_curve["r2"].iloc[-1], color="#7F77DD",
                       linestyle="--", linewidth=1,
                       label=f"Full CLIP R²={df_curve['r2'].iloc[-1]:.3f}")
            ax.set_xlabel("CLIP budget (%)")
            ax.set_ylabel("CATE R²")
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close()

        with col_r:
            st.subheader("Efficiency frontier")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(df_curve["n_clip"], df_curve["r2_gain"],
                    color="#BA7517", linewidth=2.5, marker="o", markersize=4)
            ax.set_xlabel("Ads processed with CLIP")
            ax.set_ylabel("R² gain over text-only")
            st.pyplot(fig)
            plt.close()

        st.subheader("Budget summary table")
        st.dataframe(
            df_curve[["budget_pct", "n_clip", "r2", "r2_gain", "efficiency"]]
            .rename(columns={
                "budget_pct": "Budget (%)",
                "n_clip":     "Ads with CLIP",
                "r2":         "CATE R²",
                "r2_gain":    "R² gain",
                "efficiency": "Efficiency (%)"
            }),
            use_container_width=True
        )

    st.divider()

    # ── Router evaluation ────────────────────────────────────
    st.subheader("Routing classifier performance")

    routing_eval_path = OUTPUTS_DIR / "routing_eval.json"
    if routing_eval_path.exists():
        with open(routing_eval_path) as f:
            routing_eval = json.load(f)

        rows = []
        for budget, metrics in routing_eval.items():
            rows.append({
                "Budget (%)":  metrics["budget_pct"],
                "Ads to CLIP": metrics["n_clip"],
                "Recall":      metrics["recall"],
                "Precision":   metrics["precision"],
                "F1":          metrics["f1"]
            })
        df_eval = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(df_eval["Budget (%)"], df_eval["Recall"],
                color="#1D9E75", marker="o", label="Recall")
        ax.plot(df_eval["Budget (%)"], df_eval["Precision"],
                color="#D85A30", marker="o", label="Precision")
        ax.plot(df_eval["Budget (%)"], df_eval["F1"],
                color="#7F77DD", marker="o", label="F1")
        ax.set_xlabel("CLIP budget (%)")
        ax.set_ylabel("Score")
        ax.set_title("Router precision / recall / F1 vs budget")
        ax.legend()
        st.pyplot(fig)
        plt.close()

        st.dataframe(df_eval, use_container_width=True)

    # ── Key insight ──────────────────────────────────────────
    st.divider()
    st.subheader("Key insight")
    st.info(
        "**CLIP is essential for causal estimation, not conversion prediction.**\n\n"
        "Text-only features achieve R²=0.997 for mean ICTR prediction — "
        "leaving no room for visual features. But for CATE estimation, "
        "text-only R²=0.317 vs full CLIP R²=0.792. "
        "Visual features explain *why* some ads convert better, "
        "even when they can't predict *whether* an ad converts.\n\n"
        "The adaptive router (AUC=0.933) identifies which ads need visual "
        "analysis with 100% precision at 10% budget, enabling 10× compute "
        "savings while preserving causal attribution quality."
    )
"""
Week 6: Temporal trigger visualization
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT_DIR      = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR   = ROOT_DIR / "outputs"

def plot_all():
    df_master = pd.read_parquet(PROCESSED_DIR / "dataset_causal.parquet")
    df_aligned = pd.read_parquet(PROCESSED_DIR / "ictr_aligned.parquet")
    df_peaks   = pd.read_parquet(PROCESSED_DIR / "conversion_peaks.parquet")

    with open(OUTPUTS_DIR / "curve_comparisons.json") as f:
        curve_results = json.load(f)

    time_cols = [c for c in df_aligned.columns if c.startswith("t")]
    x = np.linspace(0, 100, len(time_cols))  # 0-100% of ad duration

    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Plot 1: Peak position distribution ──────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bins = pd.cut(df_peaks["peak_relative_pos"],
                  bins=20, labels=False)
    bin_centers = np.linspace(0, 100, 20)
    counts = [((bins == i).sum()) for i in range(20)]
    ax1.bar(bin_centers, counts, width=4.5,
            color="#1D9E75", alpha=0.7, edgecolor="white")
    ax1.axvline(75, color="#D85A30", linestyle="--",
                linewidth=1.5, label="75% mark")
    ax1.set_xlabel("Peak position (% of ad duration)")
    ax1.set_ylabel("Number of ads")
    ax1.set_title("Conversion peak position distribution")
    ax1.legend(fontsize=9)

    # ── Plot 2: Trigger type pie ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ttype = df_peaks["trigger_type"].value_counts()
    colors = ["#888780", "#1D9E75", "#7F77DD", "#D85A30", "#BA7517"]
    wedges, texts, autotexts = ax2.pie(
        ttype.values,
        labels=ttype.index,
        autopct="%1.1f%%",
        colors=colors[:len(ttype)],
        startangle=90
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax2.set_title("Content type before conversion peak")

    # ── Plot 3-6: Treated vs Control ICTR curves ─────────────
    plot_treatments = [
        "T_promo_first_5s",
        "T_cta_early",
        "T_cta_late",
        "T_peak_in_last_quarter"
    ]
    colors_tc = {"treated": "#1D9E75", "control": "#888780"}

    for idx, t_name in enumerate(plot_treatments):
        row = 1 + idx // 2
        col = idx % 2
        ax  = fig.add_subplot(gs[row, col])

        # 找对应的 curve result
        res = next((r for r in curve_results
                    if r["treatment"] == t_name), None)
        if res is None:
            ax.set_visible(False)
            continue

        mean_t = np.array(res["mean_treated"])
        mean_c = np.array(res["mean_control"])

        # 95% CI（从 aligned data 重新算）
        df_m = df_aligned.merge(
            df_master[["ad_id", t_name]], on="ad_id"
        )
        treated_vals = df_m[df_m[t_name] == 1][time_cols].values
        control_vals = df_m[df_m[t_name] == 0][time_cols].values

        def ci95(arr):
            se = arr.std(axis=0) / np.sqrt(len(arr))
            return 1.96 * se

        ci_t = ci95(treated_vals)
        ci_c = ci95(control_vals)

        ax.plot(x, mean_t, color=colors_tc["treated"],
                linewidth=2, label=f"Treated (n={res['n_treated']})")
        ax.fill_between(x, mean_t - ci_t, mean_t + ci_t,
                        color=colors_tc["treated"], alpha=0.15)

        ax.plot(x, mean_c, color=colors_tc["control"],
                linewidth=2, label=f"Control (n={res['n_control']})",
                linestyle="--")
        ax.fill_between(x, mean_c - ci_c, mean_c + ci_c,
                        color=colors_tc["control"], alpha=0.15)

        # 标注显著时间窗口
        for w in res["sig_windows"]:
            ax.axvspan(x[w] - 2.5, x[w] + 2.5,
                       alpha=0.12, color="#D85A30")

        ax.set_xlabel("Ad progress (%)")
        ax.set_ylabel("Mean ICTR")
        ax.set_title(t_name.replace("T_", "").replace("_", " ").title())
        ax.legend(fontsize=8)

    plt.suptitle("Temporal trigger analysis — AdLens",
                 fontsize=14, y=1.01)
    out = OUTPUTS_DIR / "temporal_triggers.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f" Saved {out}")
    plt.show()


    # ── Summary table ────────────────────────────────────────
    print("\n=== Temporal treatment summary ===")
    rows = []
    for res in curve_results:
        t = res["treatment"]
        mean_t = np.array(res["mean_treated"]).mean()
        mean_c = np.array(res["mean_control"]).mean()
        lift   = (mean_t - mean_c) / (mean_c + 1e-9) * 100
        rows.append({
            "treatment":      t,
            "n_treated":      res["n_treated"],
            "n_control":      res["n_control"],
            "mean_ictr_treated": round(mean_t, 5),
            "mean_ictr_control": round(mean_c, 5),
            "lift_%":         round(lift, 1),
            "sig_windows":    len(res["sig_windows"])
        })

    df_summary = pd.DataFrame(rows).sort_values("lift_%", ascending=False)
    print(df_summary.to_string(index=False))
    df_summary.to_csv(OUTPUTS_DIR / "temporal_summary.csv", index=False)
    print("\n Saved temporal_summary.csv")


if __name__ == "__main__":
    plot_all()

'''
CTA timing: almost no impact
T_peak_in_last_quarter kuft = +349% -> correlation not causation
Counterfactual: T_promo_first_5s lift = -19% 4 significant windows; T_promo_first_ss lift = -14.7% 2 significant windows -> early promo has negative impact.
'''
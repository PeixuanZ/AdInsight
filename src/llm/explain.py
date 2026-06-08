"""
Week 7: LLM Explanation Layer
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import requests

ROOT_DIR      = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR   = ROOT_DIR / "outputs"

def call_llm(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


def build_ad_context(ad_id: str,
                     df_master: pd.DataFrame,
                     df_peaks: pd.DataFrame) -> dict:
    """构建单个广告的完整上下文供 LLM 分析。"""
    row = df_master[df_master["ad_id"] == ad_id].iloc[0]
    peak = df_peaks[df_peaks["ad_id"] == ad_id]

    ctx = {
        "ad_id":         ad_id,
        "full_text":     row.get("full_text", ""),
        "duration_sec":  float(row.get("duration_sec", 0)),
        "mean_ictr":     float(row.get("mean_ictr", 0)),
        "max_ictr":      float(row.get("max_ictr", 0)),
        "peak_sec":      float(peak["peak_sec"].iloc[0]) if len(peak) else -1,
        "peak_relative_pos": float(peak["peak_relative_pos"].iloc[0]) if len(peak) else -1,
        "trigger_type":  peak["trigger_type"].iloc[0] if len(peak) else "unknown",
        "trigger_text":  peak["trigger_text"].iloc[0] if len(peak) else "",
        "n_segments":    int(row.get("n_segments", 0)),
        "speech_rate":   float(row.get("speech_rate", 0)),
        "has_promo":     int(row.get("has_promo", 0)),
        "has_cta":       int(row.get("has_cta", 0)),
        "T_promo_first_5s": int(row.get("T_promo_first_5s", 0)),
        "T_peak_in_last_quarter": int(row.get("T_peak_in_last_quarter", 0)),
        "cate":          float(row.get("cate", 0)) if "cate" in row.index else None,
    }
    return ctx


def explain_single_ad(ctx: dict) -> str:
    """用 llama 解释单个广告的转化表现。"""
    prompt = f"""You are an e-commerce advertising analyst specializing in causal inference.

Analyze this video ad's conversion performance and provide actionable insights.

Ad data:
- Full transcript: {ctx['full_text'][:300]}
- Duration: {ctx['duration_sec']:.0f}s
- Mean ICTR (conversion rate): {ctx['mean_ictr']:.4f}
- Peak ICTR: {ctx['max_ictr']:.4f}
- Conversion peak at: {ctx['peak_relative_pos']*100:.0f}% of ad duration ({ctx['peak_sec']:.0f}s)
- Content before peak: "{ctx['trigger_text']}"
- Trigger type: {ctx['trigger_type']}
- Has promotional content: {'Yes' if ctx['has_promo'] else 'No'}
- Has CTA: {'Yes' if ctx['has_cta'] else 'No'}
- Promo in first 5s: {'Yes' if ctx['T_promo_first_5s'] else 'No'}
- Peak in last quarter: {'Yes' if ctx['T_peak_in_last_quarter'] else 'No'}
{f"- Causal effect of promo (CATE): {ctx['cate']:.5f}" if ctx['cate'] is not None else ""}

Provide a concise analysis (3-4 sentences) covering:
1. What content pattern likely triggered the conversion peak
2. Whether this ad follows effective or ineffective timing patterns
3. One specific actionable recommendation

Respond in English. Be specific and data-driven."""

    response = call_llm(
        prompt
    )
    return response


def generate_portfolio_insight(df_master: pd.DataFrame,
                                curve_results: list) -> str:
    """生成整个广告组合的宏观洞察。"""
    # 统计摘要
    peak_last_q = df_master["T_peak_in_last_quarter"].mean() * 100
    promo_first5 = df_master["T_promo_first_5s"].mean() * 100
    mean_ictr_overall = df_master["mean_ictr"].mean()

    # 找 lift 最大的 treatment
    summary_path = OUTPUTS_DIR / "temporal_summary.csv"
    df_summary = pd.read_csv(summary_path)
    top_treatment = df_summary.iloc[0]

    prompt = f"""You are a senior e-commerce advertising strategist.

Based on causal analysis of {len(df_master)} video ads, here are the key findings:

Dataset statistics:
- Total ads analyzed: {len(df_master)}
- Average conversion rate (ICTR): {mean_ictr_overall:.4f}
- {peak_last_q:.1f}% of ads have conversion peak in last 25% of duration
- {promo_first5:.1f}% of ads open with promotional content in first 5s

Causal findings:
- Ads with conversion peak in last quarter: +349% lift in mean ICTR
- Early promotional content (first 5s): -19% lift (NEGATIVE effect)
- Early CTA (first half): -5% lift (slight negative)
- Late CTA (second half): +0.4% lift (neutral)
- 73.6% of conversion peaks are triggered by unclassified content ("other")

Write a strategic memo (5-6 sentences) for an e-commerce marketing team covering:
1. The single most important timing insight
2. What the "other" trigger type suggests about what drives conversion
3. Recommended ad structure based on causal evidence
4. One hypothesis for future A/B testing

Be direct and actionable. Write in English."""

    response = call_llm(
        prompt
    )
    return response


def main():
    df_master = pd.read_parquet(PROCESSED_DIR / "dataset_causal.parquet")
    df_peaks  = pd.read_parquet(PROCESSED_DIR / "conversion_peaks.parquet")

    with open(OUTPUTS_DIR / "curve_comparisons.json") as f:
        curve_results = json.load(f)

    # ── Part 1: 分析高/低转化率代表性广告各 5 个 ──────────────
    print("=== Part 1: Individual ad explanations ===")

    top5    = df_master.nlargest(5, "mean_ictr")["ad_id"].astype(str).tolist()
    bottom5 = df_master.nsmallest(5, "mean_ictr")["ad_id"].astype(str).tolist()

    ad_explanations = []
    for label, ids in [("high_performer", top5),
                        ("low_performer", bottom5)]:
        for ad_id in ids:
            print(f"  Explaining {label} ad {ad_id}...")
            ctx  = build_ad_context(ad_id, df_master, df_peaks)
            text = explain_single_ad(ctx)
            ad_explanations.append({
                "ad_id":       ad_id,
                "label":       label,
                "mean_ictr":   ctx["mean_ictr"],
                "explanation": text
            })
            print(f"    → {text[:120]}...")

    df_explanations = pd.DataFrame(ad_explanations)
    df_explanations.to_csv(OUTPUTS_DIR / "ad_explanations.csv",
                           index=False)
    print(f"\n✓ Saved ad_explanations.csv")

    # ── Part 2: 整体组合洞察 ──────────────────────────────────
    print("\n=== Part 2: Portfolio insight ===")
    insight = generate_portfolio_insight(df_master, curve_results)
    print("\n" + insight)

    with open(OUTPUTS_DIR / "portfolio_insight.txt", "w") as f:
        f.write(insight)
    print("\n✓ Saved portfolio_insight.txt")

    # ── Part 3: 把解释合并回主数据集 ─────────────────────────
    df_master["ad_id"] = df_master["ad_id"].astype(str)
    df_exp_merge = df_explanations[["ad_id", "explanation"]]
    df_master = df_master.merge(df_exp_merge, on="ad_id", how="left")
    df_master.to_parquet(PROCESSED_DIR / "dataset_final.parquet",
                         index=False)
    print("✓ Saved dataset_final.parquet")


if __name__ == "__main__":
    main()
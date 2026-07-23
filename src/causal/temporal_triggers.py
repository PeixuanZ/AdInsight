"""
Week 6: Temporal Trigger Discovery
找出广告中哪些时间点/内容触发了转化高峰
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from scipy.signal import find_peaks
from statsmodels.stats.multitest import multipletests

ROOT_DIR        = Path(__file__).parent.parent.parent
RAW_DIR         = ROOT_DIR / "data" / "raw"
PROCESSED_DIR   = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR     = ROOT_DIR / "outputs"
ICTR_DIR        = RAW_DIR / "ictr" / "ictr"
TRANSCRIPTS_DIR = RAW_DIR / "transcripts" / "transcripts"
OUTPUTS_DIR.mkdir(exist_ok=True)

PROMO_KW   = ["限时", "限量", "秒杀", "抢购", "福利", "赠", "礼盒", "特惠", "活动"]
CTA_KW     = ["点击", "下单", "购买", "链接", "买", "带走", "加购"]
PRICE_KW   = ["元", "折", "价", "优惠", "券", "减", "免", "返"]
URGENCY_KW = ["最后", "仅剩", "今天", "立即", "马上", "倒计时"]


# ── Part 1: 找每个广告的转化峰值 ──────────────────────────────

def find_conversion_peaks(ad_id: str) -> dict:
    """
    从逐秒 ICTR 里找转化峰值时间点。
    返回峰值时间、峰值大小、峰值前的 segment 内容。
    """
    ictr_path = ICTR_DIR / f"{ad_id}.csv"
    trans_path = TRANSCRIPTS_DIR / f"{ad_id}.json"

    if not ictr_path.exists():
        return None

    df_i = pd.read_csv(ictr_path)
    ictr = df_i["ictr"].values
    sec  = df_i["sec"].values
    duration = sec.max()

    # scipy 找峰值（prominence 过滤掉小波动）
    peaks, props = find_peaks(
        ictr,
        prominence=ictr.std() * 0.5,
        distance=2
    )

    if len(peaks) == 0:
        # 没有明显峰值，用最大值点
        peaks = [ictr.argmax()]

    # 主峰（prominence 最大）
    if len(peaks) > 1 and "prominences" in props:
        main_peak_idx = peaks[props["prominences"].argmax()]
    else:
        main_peak_idx = peaks[0]

    peak_sec = float(sec[main_peak_idx])
    peak_val = float(ictr[main_peak_idx])

    # 找峰值前 3 秒内的 transcript segment
    trigger_segments = []
    if trans_path.exists():
        with open(trans_path) as f:
            t = json.load(f)
        for seg in t.get("segments", []):
            if peak_sec - 3 <= seg["end"] <= peak_sec + 1:
                trigger_segments.append(seg["text"])

    # 峰值前内容的关键词类型
    trigger_text = " ".join(trigger_segments)
    trigger_type = "other"
    if any(k in trigger_text for k in CTA_KW):
        trigger_type = "cta"
    elif any(k in trigger_text for k in PROMO_KW):
        trigger_type = "promo"
    elif any(k in trigger_text for k in PRICE_KW):
        trigger_type = "price"
    elif any(k in trigger_text for k in URGENCY_KW):
        trigger_type = "urgency"

    return {
        "ad_id":              ad_id,
        "peak_sec":           peak_sec,
        "peak_val":           peak_val,
        "peak_relative_pos":  round(peak_sec / duration, 3) if duration > 0 else 0,
        "n_peaks":            len(peaks),
        "trigger_text":       trigger_text[:200],
        "trigger_type":       trigger_type,
        "duration":           float(duration),
        "mean_ictr":          float(ictr.mean()),
        "ictr_at_peak_ratio": round(peak_val / (ictr.mean() + 1e-9), 2)
    }


# ── Part 2: 对齐 ICTR 时间序列（归一化到 0-1）───────────────────

def align_ictr_series(ad_ids: list, n_bins: int = 20) -> pd.DataFrame:
    """
    把所有广告的 ICTR 时间序列归一化到相同长度（n_bins 个时间点）。
    用于计算平均转化曲线和分组对比。
    """
    rows = []
    for aid in tqdm(ad_ids, desc="Aligning ICTR"):
        path = ICTR_DIR / f"{aid}.csv"
        if not path.exists():
            continue
        df_i = pd.read_csv(path)
        ictr = df_i["ictr"].values
        # 线性插值到 n_bins 个点
        x_old = np.linspace(0, 1, len(ictr))
        x_new = np.linspace(0, 1, n_bins)
        ictr_resampled = np.interp(x_new, x_old, ictr)
        rows.append([aid] + list(ictr_resampled))

    cols = ["ad_id"] + [f"t{i:02d}" for i in range(n_bins)]
    return pd.DataFrame(rows, columns=cols)


# ── Part 3: 分组对比——timing treatment 对转化曲线的影响 ──────────

def compare_ictr_curves(df_aligned: pd.DataFrame,
                        df_master: pd.DataFrame,
                        treatment: str) -> dict:
    """
    对比 treated vs control 的平均 ICTR 曲线。
    返回每个时间点的 t-test p-value，找出显著差异的时间窗口。
    """
    time_cols = [c for c in df_aligned.columns if c.startswith("t")]

    df_merged = df_aligned.merge(
        df_master[["ad_id", treatment]], on="ad_id"
    )

    treated = df_merged[df_merged[treatment] == 1][time_cols].values
    control = df_merged[df_merged[treatment] == 0][time_cols].values

    if len(treated) < 10 or len(control) < 10:
        return None

    mean_treated = treated.mean(axis=0)
    mean_control = control.mean(axis=0)

    p_values = []
    for i in range(len(time_cols)):
        _, p = stats.ttest_ind(treated[:, i], control[:, i])
        p_values.append(p)

    return {
        "treatment":     treatment,
        "n_treated":     len(treated),
        "n_control":     len(control),
        "mean_treated":  mean_treated.tolist(),
        "mean_control":  mean_control.tolist(),
        "p_values":      p_values,
        "sig_windows":   [i for i, p in enumerate(p_values) if p < 0.05]
        # sig_windows_fdr added later in main() — FDR correction needs
        # to see p-values pooled across ALL treatments x time bins at
        # once, not one treatment's 20 bins in isolation
    }


# ── Main ──────────────────────────────────────────────────────

def main():
    df_master = pd.read_parquet(PROCESSED_DIR / "dataset_causal.parquet")
    ad_ids    = df_master["ad_id"].astype(str).tolist()

    # Part 1: 找所有广告的转化峰值
    print("=== Part 1: Finding conversion peaks ===")
    peak_rows = []
    for aid in tqdm(ad_ids, desc="Finding peaks"):
        res = find_conversion_peaks(aid)
        if res:
            peak_rows.append(res)

    df_peaks = pd.DataFrame(peak_rows)
    print(f"\nPeak summary ({len(df_peaks)} ads):")
    print(df_peaks[["peak_sec", "peak_relative_pos",
                     "n_peaks", "trigger_type"]].describe())

    print("\nTrigger type distribution:")
    print(df_peaks["trigger_type"].value_counts())

    print("\nPeak position distribution:")
    bins = pd.cut(df_peaks["peak_relative_pos"],
                  bins=[0, 0.25, 0.5, 0.75, 1.0],
                  labels=["0-25%", "25-50%", "50-75%", "75-100%"])
    print(bins.value_counts().sort_index())

    df_peaks.to_parquet(PROCESSED_DIR / "conversion_peaks.parquet",
                        index=False)
    print("\n✓ Saved conversion_peaks.parquet")

    # Part 2: 对齐 ICTR 曲线
    print("\n=== Part 2: Aligning ICTR series ===")
    df_aligned = align_ictr_series(ad_ids, n_bins=20)
    df_aligned.to_parquet(PROCESSED_DIR / "ictr_aligned.parquet",
                          index=False)
    print(f"✓ Saved ictr_aligned.parquet  shape={df_aligned.shape}")

    # Part 3: 分组对比
    print("\n=== Part 3: Treatment curve comparison ===")
    treatments_to_compare = [
        "T_promo_first_3s", "T_promo_first_5s",
        "T_cta_early", "T_cta_late",
        "T_peak_in_last_quarter"
    ]

    curve_results = []
    for t in treatments_to_compare:
        if t not in df_master.columns:
            print(f"  skip {t} (not in dataset)")
            continue
        res = compare_ictr_curves(df_aligned, df_master, t)
        if res is None:
            print(f"  skip {t} (too few samples)")
            continue
        n_sig = len(res["sig_windows"])
        print(f"  {t}: {res['n_treated']} treated, "
              f"{res['n_control']} control, "
              f"{n_sig} significant time windows (raw p<0.05)")
        curve_results.append(res)

    # ── FDR correction, pooled across ALL treatments x time bins ──
    # We're really running one big batch here: 5 treatments x 20 time
    # bins = up to 100 t-tests. At raw p<0.05 that's ~5 false positives
    # expected by chance alone even if nothing were real. Pool every
    # p-value across every treatment and correct once, then map back.
    all_p = []
    index_map = []  # (result_idx, bin_idx) for each p-value, in order
    for ri, res in enumerate(curve_results):
        for bi, p in enumerate(res["p_values"]):
            all_p.append(p)
            index_map.append((ri, bi))

    if all_p:
        reject, q_values, _, _ = multipletests(all_p, alpha=0.05, method="fdr_bh")
        for res in curve_results:
            res["q_values"] = [None] * len(res["p_values"])
            res["sig_windows_fdr"] = []
        for (ri, bi), q, rej in zip(index_map, q_values, reject):
            curve_results[ri]["q_values"][bi] = float(q)
            if rej:
                curve_results[ri]["sig_windows_fdr"].append(bi)

        print(f"\n=== Multiple-testing correction (pooled, "
              f"{len(all_p)} tests, Benjamini-Hochberg FDR) ===")
        for res in curve_results:
            n_raw = len(res["sig_windows"])
            n_fdr = len(res["sig_windows_fdr"])
            print(f"  {res['treatment']}: raw={n_raw} windows -> "
                  f"FDR-corrected={n_fdr} windows")

    with open(OUTPUTS_DIR / "curve_comparisons.json", "w") as f:
        json.dump(curve_results, f, indent=2)
    print("\n✓ Saved curve_comparisons.json")

    # 合并峰值信息到 master dataset
    df_out = df_master.merge(
        df_peaks[["ad_id", "peak_sec", "peak_relative_pos",
                  "n_peaks", "trigger_type", "ictr_at_peak_ratio"]],
        on="ad_id", how="left"
    )
    df_out.to_parquet(PROCESSED_DIR / "dataset_causal.parquet",
                      index=False)
    print("Updated dataset_causal.parquet with peak info")


if __name__ == "__main__":
    main()

'''
Key Observations
1. Peak position: 47% in the last 25% (75%-100%) -> most ads is 'watcher longer, then buy more'
2. Trugger type: 73% are other -> triggers are not the defined key words
3. T_peak_in_last_quarter: 17 significant time windows
'''
"""
Week 4: Merge all features, define treatments and confounders
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def load_and_merge() -> pd.DataFrame:
    """Merge master + week2 features + clip features."""
    df_master   = pd.read_parquet(PROCESSED_DIR / "master.parquet")
    df_week2    = pd.read_parquet(PROCESSED_DIR / "features_week2.parquet")
    df_clip     = pd.read_parquet(PROCESSED_DIR / "features_clip.parquet")

    df = df_master.merge(df_week2, on="ad_id", how="left")
    df = df.merge(df_clip,   on="ad_id", how="left")

    print(f"Merged shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


def define_treatments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary treatment variables — timing-based + content-based
    """
    # ── 原有 treatments ──────────────────────────
    df["T_has_promo"]   = (df["has_promo"] == 1).astype(int)
    df["T_has_cta"]     = (df["has_cta"] == 1).astype(int)
    df["T_has_price"]   = (df["has_price"] == 1).astype(int)
    df["T_has_urgency"] = (df["has_urgency"] == 1).astype(int)
    df["T_late_cta"]    = (df["cta_in_last_quarter"] == 1).astype(int)
    df["T_early_promo"] = (df["promo_in_first_half"] == 1).astype(int)

    # ── 新增 temporal treatments ─────────────────
    # Updated after week 5 analysis
    import json
    from pathlib import Path
    TRANSCRIPTS_DIR = Path("data/raw/transcripts/transcripts")

    PROMO_KW = ["限时", "限量", "秒杀", "抢购", "福利", "赠", "礼盒", "特惠", "活动"]
    CTA_KW   = ["点击", "下单", "购买", "链接", "买", "带走", "加购"]

    rows = []
    for aid in df["ad_id"].astype(str):
        path = TRANSCRIPTS_DIR / f"{aid}.json"
        rec  = {"ad_id": aid}

        if not path.exists():
            rec.update({
                "T_promo_first_3s": 0, "T_promo_first_5s": 0,
                "T_promo_last_5s":  0, "T_cta_early": 0,
                "T_cta_late":       0, "T_promo_cta_gap": -1,
                "T_peak_in_last_quarter": 0
            })
            rows.append(rec)
            continue

        with open(path) as f:
            t = json.load(f)

        segs     = t.get("segments", [])
        duration = segs[-1]["end"] if segs else 0

        # 找第一次出现促销词/CTA 的时间
        first_promo_t = next(
            (s["start"] for s in segs
             if any(k in s["text"] for k in PROMO_KW)), -1
        )
        first_cta_t = next(
            (s["start"] for s in segs
             if any(k in s["text"] for k in CTA_KW)), -1
        )

        # Promo timing treatments
        rec["T_promo_first_3s"] = int(0 <= first_promo_t <= 3)
        rec["T_promo_first_5s"] = int(0 <= first_promo_t <= 5)
        rec["T_promo_last_5s"]  = int(
            first_promo_t > 0 and duration > 0
            and first_promo_t >= duration - 5
        )

        # CTA timing treatments
        rec["T_cta_early"] = int(
            0 <= first_cta_t < duration / 2
        )
        rec["T_cta_late"]  = int(
            first_cta_t >= duration / 2
        )

        # 促销词到CTA的时间差（连续变量）
        if first_promo_t >= 0 and first_cta_t >= 0:
            rec["T_promo_cta_gap"] = round(first_cta_t - first_promo_t, 2)
        else:
            rec["T_promo_cta_gap"] = -1

        rows.append(rec)

    df_timing = pd.DataFrame(rows)
    df = df.merge(df_timing, on="ad_id", how="left")

    # ── T_peak_in_last_quarter（从 ICTR 计算）────
    df["T_peak_in_last_quarter"] = (
        df["peak_relative_pos"] >= 0.75
    ).astype(int)

    # ── 打印分布 ──────────────────────────────────
    all_treatments = [
        "T_has_promo", "T_has_cta", "T_has_price",
        "T_has_urgency", "T_late_cta", "T_early_promo",
        "T_promo_first_3s", "T_promo_first_5s", "T_promo_last_5s",
        "T_cta_early", "T_cta_late", "T_peak_in_last_quarter"
    ]

    print("\nTreatment distributions:")
    for t in all_treatments:
        if t in df.columns:
            n1  = df[t].sum()
            pct = n1 / len(df) * 100
            bar = "█" * int(pct / 5)
            print(f"  {t:<28} {pct:5.1f}%  {bar}")

    # T_promo_cta_gap 是连续变量，单独打印
    gap = df["T_promo_cta_gap"]
    valid = gap[gap >= 0]
    print(f"\n  T_promo_cta_gap (continuous):")
    print(f"    有效样本: {len(valid)} / {len(df)}")
    print(f"    mean={valid.mean():.1f}s  "
          f"median={valid.median():.1f}s  "
          f"range=[{valid.min():.1f}, {valid.max():.1f}]")

    return df


def define_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Outcome variables — conversion-related metrics from ICTR.
    """
    # Y1: 平均转化率（主要 outcome）
    df["Y_mean_ictr"] = df["mean_ictr"]

    # Y2: 峰值转化率
    df["Y_max_ictr"] = df["max_ictr"]

    # Y3: 后段 vs 前段转化率比值（是否越看越想买）
    # Winsorize 处理极端值
    ratio = df["late_early_ratio"].clip(
        upper=df["late_early_ratio"].quantile(0.95)
    )
    df["Y_late_early_ratio"] = ratio

    # Y4: 转化率斜率（正 = 越来越高）
    df["Y_ictr_slope"] = df["ictr_slope"]

    outcomes = ["Y_mean_ictr", "Y_max_ictr", "Y_late_early_ratio", "Y_ictr_slope"]
    print("\nOutcome summary:")
    print(df[outcomes].describe())

    return df


def define_confounders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Confounders — variables that affect both treatment and outcome.
    """
    # 视觉风格（CLIP softmax scores）
    vis_cols = [c for c in df.columns if c.startswith("vis_")
                and not c.startswith("vis_z_")]

    # 广告时长（影响能放多少内容）
    # 说话速度（影响信息密度）
    # segment 数量（信息量）
    structural_confounders = ["duration_sec", "speech_rate", "n_segments", "n_frames"]

    all_confounders = vis_cols + structural_confounders
    print(f"\nConfounders ({len(all_confounders)} total):")
    print(f"  Visual: {vis_cols}")
    print(f"  Structural: {structural_confounders}")

    return df, all_confounders


def winsorize_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Clip extreme outcome values at 99th percentile."""
    for col in ["Y_mean_ictr", "Y_max_ictr"]:
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=upper)
        print(f"  {col} clipped at {upper:.4f}")
    return df


if __name__ == "__main__":
    print("=== Step 1: Merge features ===")
    df = load_and_merge()

    print("\n=== Step 2: Define treatments ===")
    df = define_treatments(df)

    print("\n=== Step 3: Define outcomes ===")
    df = define_outcomes(df)

    print("\n=== Step 4: Define confounders ===")
    df, confounders = define_confounders(df)

    print("\n=== Step 5: Winsorize outcomes ===")
    df = winsorize_outcomes(df)

    # 删除缺失值
    key_cols = (["T_has_promo", "T_has_cta", "Y_mean_ictr"] + confounders)
    before = len(df)
    df = df.dropna(subset=key_cols)
    print(f"\nDropped {before - len(df)} rows with missing values")
    print(f"Final dataset: {df.shape}")

    # 保存
    out = PROCESSED_DIR / "dataset_causal.parquet"
    df.to_parquet(out, index=False)
    print(f"\n✓ Saved to {out}")

    # 保存 confounder 列表供后续使用
    import json
    meta = {
        "treatments": [
            "T_has_promo", "T_has_cta", "T_has_price",
            "T_early_promo",
            "T_promo_first_3s", "T_promo_first_5s", "T_promo_last_5s",
            "T_cta_early", "T_cta_late"
        ],
        "treatments_skip": ["T_has_urgency"],  # 样本太少
        "treatments_descriptive_only": ["T_peak_in_last_quarter"],
        "continuous_treatments": ["T_promo_cta_gap"],
        "outcomes":   ["Y_mean_ictr", "Y_max_ictr",
                       "Y_late_early_ratio", "Y_ictr_slope"],
        "confounders": confounders
    }
    with open(PROCESSED_DIR / "causal_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved causal_meta.json")
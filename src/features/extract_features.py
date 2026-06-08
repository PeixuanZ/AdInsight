"""
Week 2: Feature extraction from transcripts and ICTR
"""
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
ICTR_DIR = RAW_DIR / "ictr" / "ictr"
TRANSCRIPTS_DIR = RAW_DIR / "transcripts" / "transcripts"

# ── 促销关键词词典 ──────────────────────────────────────────
PRICE_KEYWORDS    = ["元", "折", "价", "优惠", "券", "减", "免", "返"]
PROMO_KEYWORDS    = ["限时", "限量", "秒杀", "抢购", "福利", "赠", "礼盒", "特惠", "活动"]
URGENCY_KEYWORDS  = ["最后", "仅剩", "今天", "立即", "马上", "倒计时", "截止"]
PRODUCT_KEYWORDS  = ["成分", "效果", "功效", "配方", "专利", "认证"]
CTA_KEYWORDS      = ["点击", "下单", "购买", "链接", "买", "带走", "加购"]


def extract_keyword_features(text: str) -> dict:
    """Count keyword hits per category."""
    if not isinstance(text, str) or not text:
        return {
            "n_price_kw": 0, "n_promo_kw": 0, "n_urgency_kw": 0,
            "n_product_kw": 0, "n_cta_kw": 0,
            "has_price": 0, "has_promo": 0, "has_urgency": 0,
            "has_cta": 0
        }
    return {
        "n_price_kw":   sum(text.count(k) for k in PRICE_KEYWORDS),
        "n_promo_kw":   sum(text.count(k) for k in PROMO_KEYWORDS),
        "n_urgency_kw": sum(text.count(k) for k in URGENCY_KEYWORDS),
        "n_product_kw": sum(text.count(k) for k in PRODUCT_KEYWORDS),
        "n_cta_kw":     sum(text.count(k) for k in CTA_KEYWORDS),
        "has_price":    int(any(k in text for k in PRICE_KEYWORDS)),
        "has_promo":    int(any(k in text for k in PROMO_KEYWORDS)),
        "has_urgency":  int(any(k in text for k in URGENCY_KEYWORDS)),
        "has_cta":      int(any(k in text for k in CTA_KEYWORDS)),
    }


def extract_temporal_features(segments: list, duration: float) -> dict:
    """Extract timing-based features from transcript segments."""
    if not segments:
        return {
            "speech_rate": 0, "first_promo_time": -1,
            "first_cta_time": -1, "promo_in_first_half": 0,
            "cta_in_last_quarter": 0, "segment_density": 0
        }

    # 说话速度（字/秒）
    total_chars = sum(len(s["text"]) for s in segments)
    speech_rate = total_chars / duration if duration > 0 else 0

    # 
    first_promo_time = -1
    first_cta_time   = -1
    for seg in segments:
        text = seg["text"]
        t    = seg["start"]
        if first_promo_time < 0 and any(k in text for k in PROMO_KEYWORDS):
            first_promo_time = t
        if first_cta_time < 0 and any(k in text for k in CTA_KEYWORDS):
            first_cta_time = t

    return {
        "speech_rate":          round(speech_rate, 3),
        "first_promo_time":     first_promo_time,
        "first_cta_time":       first_cta_time,
        "promo_in_first_half":  int(0 < first_promo_time < duration / 2),
        "cta_in_last_quarter":  int(first_cta_time > duration * 0.75),
        "segment_density":      round(len(segments) / duration, 3) if duration > 0 else 0,
    }


def extract_ictr_features(df_ictr: pd.DataFrame) -> dict:
    """Extract outcome and shape features from per-second ICTR."""
    ictr = df_ictr["ictr"].values
    sec  = df_ictr["sec"].values
    duration = sec.max()

    # Peak timing
    peak_idx = ictr.argmax()
    peak_sec  = sec[peak_idx]

    # Early vs late conversion
    mid = len(ictr) // 2
    early_ictr = ictr[:mid].mean()
    late_ictr  = ictr[mid:].mean()

    # ICTR slope (linear trend)
    if len(ictr) > 1:
        slope = np.polyfit(sec, ictr, 1)[0]
    else:
        slope = 0.0

    return {
        "peak_ictr_sec":      float(peak_sec),
        "peak_ictr_val":      float(ictr.max()),
        "peak_relative_pos":  round(float(peak_sec) / duration, 3) if duration > 0 else 0,
        "early_mean_ictr":    round(float(early_ictr), 6),
        "late_mean_ictr":     round(float(late_ictr), 6),
        "late_early_ratio":   round(float(late_ictr / (early_ictr + 1e-9)), 3),
        "ictr_slope":         round(float(slope), 6),
        "ictr_std":           round(float(ictr.std()), 6),
    }


def build_feature_table() -> pd.DataFrame:
    """Main function: build full feature table for all ads."""
    df_master = pd.read_parquet(PROCESSED_DIR / "master.parquet")
    ad_ids = df_master["ad_id"].astype(str).tolist()

    rows = []
    for aid in tqdm(ad_ids, desc="Extracting features"):
        row = {"ad_id": aid}

        # ── Transcript features ──
        transcript_path = TRANSCRIPTS_DIR / f"{aid}.json"
        if transcript_path.exists():
            with open(transcript_path) as f:
                t = json.load(f)
            full_text = t.get("full_text", "")
            segments  = t.get("segments", [])
            duration  = segments[-1]["end"] if segments else 0

            row.update(extract_keyword_features(full_text))
            row.update(extract_temporal_features(segments, duration))
        else:
            row.update(extract_keyword_features(""))
            row.update(extract_temporal_features([], 0))

        # ── ICTR features ──
        ictr_path = ICTR_DIR / f"{aid}.csv"
        if ictr_path.exists():
            df_ictr = pd.read_csv(ictr_path)
            row.update(extract_ictr_features(df_ictr))

        rows.append(row)

    df_features = pd.DataFrame(rows)
    return df_features


if __name__ == "__main__":
    print("Building feature table...")
    df = build_feature_table()
    print(f"\nShape: {df.shape}")
    print(df.head(3))
    print(df.describe())

    out = PROCESSED_DIR / "features_week2.parquet"
    df.to_parquet(out, index=False)
    print(f"\n✓ Saved to {out}")

'''
Comments:
avg_n_promo_kw = 0.96: most ads have promo
avg_late_early_raio = 6.37 but max = 141 -> extreme values
ictr_slope positive or negative 
'''
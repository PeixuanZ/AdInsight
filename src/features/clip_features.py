"""
Week 3: CLIP visual features + transcript content labels
"""
import torch
import open_clip
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json

ROOT_DIR = Path(__file__).parent.parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FRAMES_DIR = RAW_DIR / "frames" / "frames"
TRANSCRIPTS_DIR = RAW_DIR / "transcripts" / "transcripts"

# ── CLIP (zero-shot labels) ──────────────────────────
VISUAL_LABELS = [
    "product close-up",
    "person using product",
    "lifestyle scene",
    "text overlay promotion",
    "before and after comparison",
    "celebrity endorsement",
    "discount price display",
    "unboxing scene",
]

TRANSCRIPT_LABELS = [
    "价格促销",      # price promotion
    "产品功效",      # product efficacy
    "限时优惠",      # limited time offer
    "用户见证",      # user testimonial
    "品牌故事",      # brand story
    "节日营销",      # holiday marketing
    "直播互动",      # live stream interaction
]


def load_clip_model(device: str = None):
    """Load CLIP model"""
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()
    return model, preprocess, tokenizer, device


def get_text_embeddings(labels: list, model, tokenizer, device: str) -> torch.Tensor:
    """Encode text labels into CLIP embeddings."""
    tokens = tokenizer(labels).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def classify_frames(ad_id: str, model, preprocess, text_features,
                    device: str, n_frames: int = 5) -> dict:
    """
    Classify sampled frames for one ad using zero-shot CLIP.
    Returns mean similarity scores per label.
    """
    frame_dir = FRAMES_DIR / ad_id
    frames = sorted(frame_dir.glob("*.jpg"))

    if not frames:
        return {f"visual_{i}": 0.0 for i in range(len(VISUAL_LABELS))}

    # Sample evenly across the ad
    indices = np.linspace(0, len(frames) - 1, min(n_frames, len(frames)), dtype=int)
    sampled = [frames[i] for i in indices]

    scores_all = []
    for fpath in sampled:
        img = preprocess(Image.open(fpath).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            img_features = model.encode_image(img)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        sims = (img_features @ text_features.T).squeeze(0).cpu().numpy()
        scores_all.append(sims)

    mean_scores = np.mean(scores_all, axis=0)
    return {
        f"clip_{label.replace(' ', '_')}": float(mean_scores[i])
        for i, label in enumerate(VISUAL_LABELS)
    }


def classify_transcript(ad_id: str, model, tokenizer,
                        transcript_text_features, device: str) -> dict:
    """Zero-shot classify transcript text against content labels."""
    path = TRANSCRIPTS_DIR / f"{ad_id}.json"
    if not path.exists():
        return {f"text_{i}": 0.0 for i in range(len(TRANSCRIPT_LABELS))}

    with open(path) as f:
        t = json.load(f)
    full_text = t.get("full_text", "")
    if not full_text:
        return {f"text_label_{i}": 0.0 for i in range(len(TRANSCRIPT_LABELS))}

    # Encode transcript (truncate to 77 tokens)
    tokens = tokenizer([full_text[:200]]).to(device)
    with torch.no_grad():
        txt_features = model.encode_text(tokens)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)

    sims = (txt_features @ transcript_text_features.T).squeeze(0).cpu().numpy()
    return {
        f"transcript_{label}": float(sims[i])
        for i, label in enumerate(TRANSCRIPT_LABELS)
    }


def extract_clip_features(sample_size: int = None) -> pd.DataFrame:
    """Main: extract CLIP features for all ads."""
    df_master = pd.read_parquet(PROCESSED_DIR / "master.parquet")
    ad_ids = df_master["ad_id"].astype(str).tolist()

    if sample_size:
        ad_ids = ad_ids[:sample_size]
        print(f"Running on sample of {sample_size} ads")

    # Load model
    model, preprocess, tokenizer, device = load_clip_model()

    # Pre-encode label embeddings
    visual_text_features     = get_text_embeddings(VISUAL_LABELS, model, tokenizer, device)
    transcript_text_features = get_text_embeddings(TRANSCRIPT_LABELS, model, tokenizer, device)

    rows = []
    for aid in tqdm(ad_ids, desc="CLIP features"):
        row = {"ad_id": aid}
        row.update(classify_frames(aid, model, preprocess,
                                   visual_text_features, device))
        row.update(classify_transcript(aid, model, tokenizer,
                                       transcript_text_features, device))
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # 先跑 50 个测试
    print("=== Test run: 50 ads ===")
    df_test = extract_clip_features(sample_size=50)
    print(df_test.shape)
    print(df_test.describe())

    # 全量
    print("\n=== Full run: 2833 ads ===")
    df_clip = extract_clip_features()

    # ── Softmax + z-score 后处理 ──────────────────────────
    visual_cols = [c for c in df_clip.columns if c.startswith("clip_")]

    def softmax(x, temp=20):
        x = np.array(x) * temp
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    df_soft = df_clip[visual_cols].apply(
        lambda row: softmax(row), axis=1, result_type="expand")
    df_soft.columns = [c.replace("clip_", "vis_") for c in visual_cols]

    df_clip["dominant_visual"] = (df_clip[visual_cols]
                                  .idxmax(axis=1)
                                  .str.replace("clip_", ""))

    df_zscore = df_clip[visual_cols].apply(
        lambda col: (col - col.mean()) / col.std())
    df_zscore.columns = [c.replace("clip_", "vis_z_") for c in visual_cols]

    df_clip = pd.concat([df_clip, df_soft, df_zscore], axis=1)

    
    out = PROCESSED_DIR / "features_clip.parquet"
    df_clip.to_parquet(out, index=False)
    print(f"Saved to {out}")

'''

'''
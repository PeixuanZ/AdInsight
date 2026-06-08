"""
Week 1: AdsInsight data loading - download and extract all zip files
"""
from huggingface_hub import hf_hub_download
import zipfile
import pandas as pd
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_adstrace():
    """Download all zip files from HuggingFace."""
    files = ["audios_16k.zip", "frames.zip", "ictr.zip", "transcripts.zip"]
    
    for fname in files:
        zip_path = RAW_DIR / fname
        if zip_path.exists():
            print(f"✓ {fname} already downloaded, skipping")
            continue
        
        print(f"Downloading {fname}...")
        hf_hub_download(
            repo_id="Xiuze/AdsTrace",
            filename=fname,
            repo_type="dataset",
            local_dir=str(RAW_DIR)
        )
        print(f"saved to {zip_path}")


def extract_all():
    """Extract all zip files."""
    files = ["audios_16k.zip", "frames.zip", "ictr.zip", "transcripts.zip"]
    
    for fname in files:
        zip_path = RAW_DIR / fname
        extract_dir = RAW_DIR / fname.replace(".zip", "")
        
        if extract_dir.exists():
            print(f"✓ {fname} already extracted, skipping")
            continue
        
        print(f"Extracting {fname}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
        print(f"  → {extract_dir}")


def inspect_structure():
    """Print folder structure and sample files."""
    for folder in ["audios_16k", "frames", "ictr", "transcripts"]:
        folder_path = RAW_DIR / folder
        if not folder_path.exists():
            print(f"✗ {folder}/ not found")
            continue
        
        all_files = list(folder_path.rglob("*"))
        sample = [f for f in all_files if f.is_file()][:5]
        
        print(f"\n{folder}/  ({len([f for f in all_files if f.is_file()])} files)")
        for f in sample:
            print(f"  {f.name}")
        print(f"  ...")


def load_ictr() -> pd.DataFrame:
    """Load the ictr (conversion rate) data - our main outcome variable."""
    ictr_dir = RAW_DIR / "ictr"
    if not ictr_dir.exists():
        raise FileNotFoundError("ictr/ not found, run download and extract first")
    
    # 找所有 csv/json/parquet 文件
    csv_files = list(ictr_dir.rglob("*.csv"))
    json_files = list(ictr_dir.rglob("*.json"))
    
    print(f"Found {len(csv_files)} csv, {len(json_files)} json files in ictr/")
    
    if csv_files:
        df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        print(f"ictr shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(df.head(3))
        return df
    elif json_files:
        df = pd.concat([pd.read_json(f) for f in json_files], ignore_index=True)
        print(f"ictr shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(df.head(3))
        return df
    else:
        all_files = list(ictr_dir.rglob("*"))
        print("Unknown file types in ictr/:")
        for f in all_files[:10]:
            print(f"  {f.name}")
        return pd.DataFrame()


if __name__ == "__main__":
    print("=== Step 1: Download ===")
    download_adstrace()
    
    print("\n=== Step 2: Extract ===")
    extract_all()
    
    print("\n=== Step 3: Inspect structure ===")
    inspect_structure()
    
    print("\n=== Step 4: Load ictr (outcome variables) ===")
    df_ictr = load_ictr()
    
    if not df_ictr.empty:
        df_ictr.to_parquet(RAW_DIR / "ictr.parquet", index=False)
        print("\n✓ Saved ictr.parquet")
# AdInsight

Stage 1. Conversion Trigger Discovery

Built AdLens, a multimodal causal-inspired attribution framework for e-commerce video advertisements using AdsTrace (2.8K ads). Leveraged CLIP, LLM-based content understanding, and Double Machine Learning to estimate the conversion impact of ad components and identify temporal conversion triggers associated with ROI and CVR improvements.

Data source: https://huggingface.co/datasets/Xiuze/AdsTrace

[x] Week 1–2：AdsTrace data cleaning and feature extraction

[x] Week 3：CLIP + Transcript content labeling

[x] Week 4： Treatment and Confounders

[x] Week 5：DML / EconML experiment

[x] Week 6：Temporal Trigger Discovery

[x] Week 7：LLM Explanation Layer

[x] Week 8：Streamlit Demo + GitHub 

Stage 2. Efficient Trigger Discovery via Adaptive Multimodal Routing

- [] Adaptive Video Routing
- [] Budget-Aware Attribution

---
AdInsight analyzes 2,833 short-form video ads from the AdsTrace dataset. It combines CLIP visual embeddings, transcript NLP, and Double Machine Learning to estimate the causal impact of ad content strategies on conversion rates (ICTR), and identifies temporal conversion triggers.

## Key Findings

- Conversion peaks cluster at the end: 47% of ads have their highest ICTR in the final 25% of duration, associated with a +349% lift in mean conversion rate
- Early promotional content hurts: ads that open with promotional keywords in the first 5 seconds show a 19% lift in ICTR vs. control
- CTA timing is neutral: whether CTA appears in the first or second half has minimal causal impact (±5%)
- 73.6% of conversion triggers are unclassified: keyword-based trigger detection misses most conversion spikes, suggesting visual or tonal cues dominate
- Heterogeneity is driven by structure: speech rate, number of frames, and duration are the top drivers of treatment effect heterogeneity, not content keywords.

## Project Structure

AdLens/
├── data/
│   ├── raw/                        # Downloaded from HuggingFace (not in Git)
│   │   ├── ictr/ictr/              # Per-second conversion rate CSVs (2,833 files)
│   │   ├── transcripts/transcripts/# Transcript JSONs with timestamps (2,833 files)
│   │   ├── frames/frames/          # Video frames per ad (59,056 JPGs, 8.3GB)
│   │   └── audios_16k/             # 16kHz WAV audio files (2,833 files)
│   └── processed/
│       ├── master.parquet          # Ad-level metadata
│       ├── features_week2.parquet  # Keyword + ICTR features
│       ├── features_clip.parquet   # CLIP visual embeddings
│       ├── dataset_causal.parquet  # Merged dataset with treatments + outcomes
│       ├── conversion_peaks.parquet# Per-ad peak detection results
│       ├── ictr_aligned.parquet    # ICTR curves resampled to 20 time bins
│       ├── dataset_with_cate.parquet # CATE estimates per ad
│       ├── dataset_final.parquet   # Final dataset with LLM explanations
│       └── causal_meta.json        # Treatment / outcome / confounder lists
│
├── src/
│   ├── ingestion/
│   │   └── load_data.py            # Download + extract AdsTrace zips
│   ├── features/
│   │   ├── extract_features.py     # Week 2: keyword + ICTR features
│   │   ├── clip_features.py        # Week 3: CLIP visual zero-shot classification
│   │   └── build_dataset.py        # Week 4: merge, define treatments + outcomes
│   └── causal/
│       ├── dml_estimate.py         # Week 5: LinearDML + CausalForest
│       ├── temporal_triggers.py    # Week 6: peak detection + curve comparison
│       └── temporal_viz.py         # Week 6: visualization
│   └── llm/
│       └── explain.py              # Week 7: LLM explanation layer (Ollama)
│
├── outputs/
│   ├── ate_results.csv             # ATE estimates per treatment
│   ├── temporal_summary.csv        # Treated vs control ICTR lift per treatment
│   ├── curve_comparisons.json      # Full ICTR curve data per treatment
│   ├── feature_importance.csv      # Causal forest heterogeneity drivers
│   ├── quartile_analysis.csv       # CATE by feature quartile
│   ├── portfolio_insight.txt       # LLM-generated strategic memo
│   ├── ad_explanations.csv         # Per-ad LLM explanations (top/bottom 5)
│   └── temporal_triggers.png       # Treated vs control ICTR curve plots
│
├── notebooks/
│   └── 01_EDA.ipynb                # Data exploration + schema inspection
│
├── app/
│   └── streamlit_app.py            # Interactive demo (4 pages)
│
├── requirements.txt
└── README.md

## Pipeline Overview

AdsInsight (HuggingFace)
        ↓
[Week 1-2] Data parsing + feature extraction
  - Per-ad ICTR statistics (mean, max, slope, peak timing)
  - Transcript keyword features (promo, CTA, price, urgency)
  - Temporal features (promo_first_3s, promo_first_5s, cta_early, cta_late)
        ↓
[Week 3] CLIP visual features
  - Zero-shot classification across 8 visual categories
  - Softmax normalization + z-score standardization
        ↓
[Week 4] Causal dataset construction
  - Treatments: 10 binary content strategy variables
  - Outcomes: Y_mean_ictr, Y_max_ictr, Y_late_early_ratio, Y_ictr_slope
  - Confounders: 8 CLIP visual scores + 4 structural features
        ↓
[Week 5] Double Machine Learning (EconML)
  - LinearDML → ATE per treatment
  - CausalForestDML → CATE per ad + feature importance
        ↓
[Week 6] Temporal trigger discovery
  - Peak detection via scipy.signal.find_peaks
  - ICTR curve alignment (20-bin resampling)
  - Treated vs control t-test per time window
        ↓
[Week 7] LLM explanation layer
  - Per-ad analysis for top/bottom performers
  - Portfolio-level strategic memo
        ↓
[Week 8] Streamlit demo

## Treatment Variables
| Variable | Description | Prevalence |
| :--- | :--- | :--- |
| `T_has_promo` | Has promotional keywords | 55.9% |
| `T_has_cta` | Has CTA keywords | 53.3% |
| `T_has_price` | Has price-related keywords | 16.2% |
| `T_early_promo` | Promo in first half | 30.6% |
| `T_promo_first_5s` | Promo in first 5 seconds | 33.8% |
| `T_promo_first_3s` | Promo in first 3 seconds | 26.1% |
| `T_promo_last_5s` | Promo in last 5 seconds | Varies |
| `T_cta_early` | CTA in first half | 41.4% |
| `T_cta_late` | CTA in second half | 11.9% |
| `T_peak_in_last_quarter` | Conversion peak in last 25% | 47.3% |

## Methods
- Double Machine Learning (DML)
Uses econml.dml.LinearDML with GradientBoostingRegressor (outcome model) and LogisticRegressionCV (treatment model). Residualizes both Y and T on confounders X, then estimates the treatment effect from the residuals. Provides ATE with 95% confidence intervals.

- Causal Forest
Uses econml.dml.CausalForestDML to estimate heterogeneous treatment effects (CATE) per ad. Feature importance identifies which confounders drive treatment effect heterogeneity.

- Temporal Trigger Discovery

  - scipy.signal.find_peaks detects ICTR peaks per ad (prominence threshold = 0.5$\sigma$)
  - ICTR series resampled to 20 bins for cross-ad alignment
  - Per-bin t-tests compare treated vs control groups
  - Content 3 seconds before each peak classified by keyword type


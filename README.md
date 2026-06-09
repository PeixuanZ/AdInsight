# AdInsight

Stage 1. Conversion Trigger Discovery

Built AdLens, a multimodal causal-inspired attribution framework for e-commerce video advertisements using AdsTrace (2.8K ads). Leveraged CLIP, LLM-based content understanding, and Double Machine Learning to estimate the conversion impact of ad components and identify temporal conversion triggers associated with ROI and CVR improvements.

Data source: https://huggingface.co/datasets/Xiuze/AdsTrace

- [x] Week 1–2：AdsTrace data cleaning and feature extraction

- [x] Week 3：CLIP + Transcript content labeling

- [x] Week 4： Treatment and Confounders

- [x] Week 5：DML / EconML experiment

- [x] Week 6：Temporal Trigger Discovery

- [x] Week 7：LLM Explanation Layer

- [x] Week 8：Streamlit Demo + GitHub 

Stage 2. Efficient Trigger Discovery via Adaptive Multimodal Routing

- [x] Adaptive Video Routing
- [x] Budget-Aware Attribution

---
AdInsight analyzes 2,833 short-form video ads from the AdsTrace dataset. It combines CLIP visual embeddings, transcript NLP, and Double Machine Learning to estimate the causal impact of ad content strategies on conversion rates (ICTR), and identifies temporal conversion triggers.

## Key Findings

- Conversion peaks cluster at the end: 47% of ads have their highest ICTR in the final 25% of duration, associated with a +349% lift in mean conversion rate
- Early promotional content hurts: ads that open with promotional keywords in the first 5 seconds show a 19% lift in ICTR vs. control
- CTA timing is neutral: whether CTA appears in the first or second half has minimal causal impact (±5%)
- 73.6% of conversion triggers are unclassified: keyword-based trigger detection misses most conversion spikes, suggesting visual or tonal cues dominate
- Heterogeneity is driven by structure: speech rate, number of frames, and duration are the top drivers of treatment effect heterogeneity, not content keywords.

## Project Structure
```
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
```
## Pipeline Overview

```
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
```
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

  - scipy.signal.find_peaks detects ICTR peaks per ad.
  - ICTR series resampled to 20 bins for cross-ad alignment.
  - Per-bin t-tests compare treated vs control groups.
  - Content 3 seconds before each peak classified by keyword type.

## Causal Confounders

| Confounder | Type | Rationale |
| :--- | :--- | :--- |
| `vis_product_close-up` | CLIP softmax | Visual style affects both content strategy and conversion |
| `vis_person_using_product` | CLIP softmax | Lifestyle vs product-focused framing |
| `vis_celebrity_endorsement` | CLIP softmax | Brand positioning signal |
| `duration_sec` | Structural | Longer ads can include more content types |
| `speech_rate` | Structural | Information density proxy |
| `n_segments` | Structural | Content complexity |
| `n_frames` | Structural | Visual editing pace |

## Limitations

- ICTR is a proxy for conversion, not ground-truth purchase data.
- Keyword-based treatment variables miss visual and tonal cues (73.6% of triggers unclassified).
- CLIP ViT-B/32 trained on English data however for Chinese text classification it might not work well.
- No randomization: causal estimates rely on unconfoundedness assumption.
- ATE not statistically significant for any treatment but heterogeneous effects dominate.

---
## Stage 2: Adaptive Multimodal Routing

### Motivation
Running CLIP on all 2,833 ads takes ~8 minutes on Apple Silicon. Can we
predict which ads need visual analysis using only cheap features?

### Module 1: Adaptive Router
Trained a routing classifier to predict which ads have high text-only
attribution residuals (i.e. need visual features for accurate CATE estimation).

**Key result**: GBM router achieves **AUC = 0.933** using only transcript
and ICTR features. At 10% budget, precision = 1.000, indicating every ad routed to
CLIP genuinely needs visual information.

| Budget | Recall | Precision | F1 |
|--------|--------|-----------|-----|
| 10% | 0.260 | 1.000 | 0.413 |
| 20% | 0.403 | 1.000 | 0.574 |
| 30% | 0.596 | 0.988 | 0.744 |
| 50% | 0.916 | 0.916 | 0.916 |

Top routing features: `mean_ictr`, `ictr_std`, `transcript_duration`,
`n_promo_kw`, `speech_rate`
| Feature | Importance | Intuition |
|---------|------------|-----------|
| `mean_ictr` | 0.343 | Ads with higher conversion rates have more complex visual patterns that text alone cannot explain |
| `ictr_std` | 0.188 | High conversion rate volatility suggests specific visual moments trigger purchases, requiring visual analysis to locate them |
| `transcript_duration` | 0.140 | Longer speech means text features are already rich — visual features provide the marginal signal |
| `n_promo_kw` | 0.087 | Many promotional keywords yet unpredictable conversion implies visuals are driving the effect, not the words |
| `speech_rate` | 0.044 | High information density means text features saturate quickly — visual context is needed to disambiguate |

### Module 2: Budget-Aware Attribution
Measured CATE prediction quality (R²) as a function of CLIP budget.

**Key result**: CLIP visual features are **essential for causal effect
estimation** but redundant for conversion rate prediction.

| Budget | CATE R² | Efficiency |
|--------|---------|------------|
| 0% (text-only) | 0.317 | 0% |
| 10% | 0.368 | 10.8% |
| 30% | 0.438 | 25.5% |
| 50% | 0.552 | 49.8% |
| 100% (full CLIP) | 0.792 | 100% |

**Negative result **: For `mean_ictr` prediction, text-only
R²=0.997 — ICTR statistics already explain 99.7% of variance, leaving
no room for visual features. CLIP only adds value when estimating
*causal effects* (CATE), not raw conversion rates.

### Files
| File | Description |
|------|-------------|
| `src/routing/adaptive_router.py` | Router training + evaluation |
| `src/routing/budget_attribution.py` | Budget sweep + curve generation |
| `outputs/routing_decisions.parquet` | Per-ad routing decisions |
| `outputs/router_meta.json` | Router AUC + feature list |
| `outputs/budget_curve.json` | R² at each budget level |
| `outputs/budget_attribution.png` | Budget-accuracy tradeoff plots |

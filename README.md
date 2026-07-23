# AdInsight

Stage 1. Conversion Trigger Discovery

Built AdInsight, a multimodal causal-inspired attribution framework for e-commerce video advertisements using AdsTrace (2.8K ads). Leveraged CLIP, LLM-based content understanding, and Double Machine Learning to estimate the conversion impact of ad components and identify temporal conversion triggers associated with ROI and CVR improvements.

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
> ⚠️ See [Validation & Limitations](#validation--limitations) below — none of these effects survived FDR correction or placebo testing. Read as exploratory/descriptive, not confirmed causal findings.

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

**Key result**: GBM router achieves **held-out test AUC = 0.932** (CV AUC = 0.931 — consistent, no overfitting) using only transcript and ICTR features.

| Budget | Recall | Precision | F1 |
|--------|--------|-----------|-----|
| 10% | 0.198 | 1.000 | 0.330 |
| 20% | 0.399 | 1.000 | 0.571 |
| 30% | 0.580 | 0.965 | 0.724 |
| 50% | 0.866 | 0.866 | 0.866 |

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
Measured CATE prediction quality (R²) as a function of CLIP budget. CATE here is the CausalForest estimate for `T_promo_first_5s` from Module 1.5 — see [Validation & Limitations](#validation--limitations) for why this specific treatment's causal validity is exploratory, not confirmed.

**Key result**: CLIP visual features substantially improve the model's ability to reconstruct the CATE estimate, at a fraction of the compute cost.

| Budget | CATE R² | Efficiency (of full-CLIP gain) |
|--------|---------|------------|
| 0% (text-only) | 0.397 | 0% |
| 10% | 0.433 | 15.0% |
| 20% | 0.466 | 28.4% |
| 30% | 0.495 | 40.5% |
| 50% | 0.538 | 58.5% |
| 100% (full CLIP) | 0.639 | 100% |

**Negative result**: For `mean_ictr` prediction directly (not CATE), text-only
R²=0.997 — ICTR statistics already explain 99.7% of variance, leaving
no room for visual features. CLIP only adds value when reconstructing
the *CATE estimate*, not raw conversion rates.

### Files
| File | Description |
|------|-------------|
| `src/routing/adaptive_router.py` | Router training + evaluation |
| `src/routing/budget_attribution.py` | Budget sweep + curve generation |
| `outputs/routing_decisions.parquet` | Per-ad routing decisions |
| `outputs/router_meta.json` | Router AUC + feature list |
| `outputs/budget_curve.json` | R² at each budget level |
| `outputs/budget_attribution.png` | Budget-accuracy tradeoff plots |


## Validation & Limitations
(Updated on 07/23/2026)

This section documents the validation methodology used across AdInsight's modules, and is transparent about what the results do and don't support. It was added after an internal audit surfaced several methodological issues (see below) — the numbers here reflect the fixes.

### Module 1 — Adaptive Router: held-out test evaluation

The router (GBM classifier deciding which ads get the expensive CLIP path) is evaluated on an 80/20 stratified train/test split, held out before any model fitting or calibration.

| Metric | Train-split CV | Held-out test |
|---|---|---|
| AUC | 0.9308 ± 0.0179 | **0.9318** (n=567) |

The held-out AUC matches the cross-validated training AUC almost exactly, indicating the router generalizes rather than overfitting. Budget-level precision/recall (below) are computed on this same untouched test split.

| Budget | Recall | Precision | F1 |
|---|---|---|---|
| 10% | 0.198 | 1.000 | 0.330 |
| 20% | 0.399 | 1.000 | 0.571 |
| 30% | 0.580 | 0.965 | 0.724 |
| 40% | 0.731 | 0.916 | 0.813 |
| 50% | 0.866 | 0.866 | 0.866 |

*Note: an earlier version of this evaluation scored the router on the same data it was fit and calibrated on (resubstitution), which inflated these numbers slightly. The held-out numbers above are the honest ones — the gap turned out to be small, confirming the router's underlying strength was real, not an artifact.*

Production routing decisions (`routing_decisions.parquet`) are generated by a router refit on 100% of the data, once the honest test-set performance above had already been established — standard practice once you trust the number, but the reported metrics themselves never touch that refit model.

### Module 1.5 — Causal validation (DML treatment effects)

Nine content-timing treatments (promo/CTA timing) were tested via `LinearDML` against `Y_mean_ictr`, with three layers of validation:

1. **Raw significance**: 0/9 treatments significant at p<0.05
2. **Multiple-testing correction** (Benjamini-Hochberg FDR, across all 9 tests): 0/9 significant after correction
3. **Placebo refutation test**: for the top 3 treatments by |ATE|, shuffling the treatment column and rerunning DML produced ATEs of similar magnitude to the real ones — i.e., the real estimates are statistically indistinguishable from noise

**Conclusion: none of the tested content-timing treatments show a causally detectable effect on mean ICTR in this dataset.** This is a genuine finding, not a failure of the pipeline — three independent checks (raw p, FDR, placebo) agree, and a parallel time-resolved analysis (below) reaches the same conclusion.

The same FDR correction was applied to `temporal_triggers.py`'s time-window t-tests (5 treatments × 20 time bins, 100 tests pooled). After correction, `T_promo_first_3s/5s` and `T_cta_early/late` mostly collapse to 0-2 significant windows, consistent with the DML null result. `T_peak_in_last_quarter` remains significant across nearly the whole curve (17/20 windows) — see the note below on why this is expected and not evidence of a real effect.

**`T_peak_in_last_quarter` was excluded from causal treatment analysis.** It's derived from `peak_relative_pos`, which comes from the same per-second ICTR curve that the outcome (`Y_mean_ictr`) is computed from — making it a descendant of the outcome rather than an upstream cause. Using it as a DML "treatment" would be a bad-control / reverse-causality error. It's retained only for descriptive analysis (e.g. "47% of ads peak in the last quarter of runtime").

### Module 2 — Budget-Aware Attribution

The routing/CLIP-budget tradeoff (30% CLIP budget captures ~40% of the accuracy gain of full CLIP usage) is measured against `cate` — the CausalForest CATE estimate for `T_promo_first_5s`, produced in Module 1.5.

**Important caveat**: per the causal validation above, `T_promo_first_5s`'s treatment effect did not survive FDR correction or the placebo test. This means Module 2's result should be read as an **engineering/systems finding** — the routing mechanism efficiently approximates a target signal with a fraction of the compute — rather than a causal claim about what drives ad performance. The routing mechanics and the OOF cross-validation methodology behind these R² numbers are sound; the label they're predicting is exploratory.

### Study design: observational, not randomized

AdInsight estimates treatment effects from historical ad performance data — ad creators weren't randomly assigned which content strategies to use. This means:

- DML's validity rests on the **unconfoundedness assumption**: that the confounders included (visual style, duration, speech rate, etc.) capture the factors that jointly affect both treatment choice and outcome. This can't be tested directly, only made more plausible with richer confounders and checked indirectly via the propensity diagnostics (`prop_min`/`prop_mean`/`prop_max`) already logged per treatment.
- The appropriate standard for "does this generalize" here isn't a train/test split (DML's `cv=3` cross-fitting already serves that role for the nuisance models) — it's refutation testing: placebo shuffles, overlap/positivity checks, and (ideally) subsample stability checks.
- **Power analysis**: at n=2,833 (α=0.05, 80% power), this dataset can detect effects of roughly Δ≥0.0021 for balanced treatments (~10% relative lift) and Δ≥0.0035 for imbalanced treatments like `T_promo_last_5s` (9% prevalence, ~16% relative lift). All observed ATEs (0.0005–0.0011) fall below these thresholds — meaning the correct conclusion is **"this dataset cannot distinguish these treatments' effects from zero, given their likely small size"**, not **"these treatments have no effect."** A live randomized A/B test would need either a much larger sample or to target treatments with hypothesized larger effect sizes to be conclusive either way.

### Summary of fixes applied (for reference)

| Issue found | Fix |
|---|---|
| Router evaluated on data it was fit/calibrated on | Added 80/20 held-out split; router reports test-set metrics, refits on full data only for production after |
| No multiple-testing correction across 9 DML treatments | Added Benjamini-Hochberg FDR correction |
| No multiple-testing correction across 100 time-bin t-tests | Added pooled FDR correction across all treatments × bins |
| No refutation check on DML estimates | Added placebo test (shuffled treatment) for top treatments |
| `T_peak_in_last_quarter` used as a causal treatment despite being derived from the outcome | Moved to descriptive-only use |
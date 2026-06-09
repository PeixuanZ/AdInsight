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
- Early promotional content hurts: ads that open with promotional keywords in the first 5 seconds show a −19% lift in ICTR vs. control
- CTA timing is neutral: whether CTA appears in the first or second half has minimal causal impact (±5%)
- 73.6% of conversion triggers are unclassified: keyword-based trigger detection misses most conversion spikes, suggesting visual or tonal cues dominate
- Heterogeneity is driven by structure: speech rate, number of frames, and duration are the top drivers of treatment effect heterogeneity — not content keywords


# LLM-Enhanced Dynamic Graph Networks for Conversion Rate Prediction

> UCL MSc Dissertation — Knowledge, Information and Data Science (KIDS)
> Liu Yize | Supervisor: Dr. Arabella Sinclair | 2025–2026

---

## Overview

This dissertation investigates whether large language model (LLM) representations can replace traditional ID-based embeddings in dynamic graph networks for conversion rate (CVR) prediction, while simultaneously mitigating the delayed feedback bias inherent in real-world advertising systems.

The core hypothesis is that LLM-derived semantic embeddings — encoding rich user profile and item attribute information — can improve both predictive accuracy and label reliability, particularly for sparse users whose interaction histories are insufficient for ID-based learning.

---

## Research Questions

| # | Research Question |
|---|---|
| RQ1 | Can LLM-derived text representations replace ID-based embeddings in dynamic graph networks for CVR prediction, and what performance gains do they yield? |
| RQ2 | Given a user profile (demographics, past purchase behaviour) encoded via LLM, can we better predict future purchase behaviour — and does this reduce the false-negative noise caused by delayed feedback? |
| RQ3 | What is the computational trade-off between the performance gains and the overhead introduced by incorporating LLM representations? |

---

## Proposed Architecture

```
Text Features (item/user descriptions)
        │
        ▼
┌─────────────────┐
│   LLM Encoder   │  (BERT / LLaMA)
│  Semantic Embs  │  replaces ID embeddings
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Dynamic Graph  │  (DGSR-based)
│  Neighbour Agg  │  long-term + short-term attention
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Delayed Feedback Layer │  (DEFER / FSIW-style loss)
│  False-negative correct │
└────────┬────────────────┘
         │
         ▼
    CVR Prediction
```

---

## Dataset

**Primary: Ali-CCP** (Alibaba Click and Conversion Prediction)

| Metric | Value |
|---|---|
| Source | Alibaba Taobao recommendation logs |
| Total impressions | ~1 billion |
| Total clicks | ~84 million |
| Total conversions | ~1.8 million (~2% CVR) |
| Users / Items | ~4 million each |
| Time span | 8 consecutive days |
| Features | User profile (age, gender, occupation), item attributes, context |

**Secondary: Criteo Conversion Log** — 16M clicks, 30-day window with real delay timestamps (used for delayed feedback benchmarking).

---

## Baselines

| Category | Baselines |
|---|---|
| Delayed Feedback | DFM, ES-DFM, FSIW, DEFER |
| Graph Recommendation | LightGCN, TGN, DGSR |
| LLM-Enhanced | UniSRec, RLMRec |

---

## Repository Structure

```
dissertation-cvr-llm/
├── data/
│   ├── README.md          # Download instructions for Ali-CCP & Criteo
│   └── preprocessing/     # Data cleaning and feature engineering scripts
├── src/
│   ├── models/
│   │   ├── llm_encoder.py       # LLM embedding module
│   │   ├── dynamic_graph.py     # DGSR-based graph network
│   │   └── df_correction.py     # Delayed feedback correction loss
│   ├── baselines/               # Baseline model implementations
│   └── utils/                   # Shared utilities
├── notebooks/                   # Exploratory analysis and visualisation
├── experiments/                 # Config files and experiment logs
├── results/                     # Evaluation outputs and figures
├── requirements.txt
└── README.md
```

---

## Progress

- [x] Literature review — Delayed Feedback (DFM, ES-DFM, FSIW, DEFER)
- [x] Literature review — Graph Networks (LightGCN, TGN, DGSR)
- [ ] Literature review — LLM Recommendation (BERT4Rec, UniSRec, RLMRec, TALLRec)
- [ ] Dataset acquisition and preprocessing
- [ ] Baseline implementation
- [ ] Model implementation
- [ ] Experiments and evaluation
- [ ] Dissertation write-up

---

## Dependencies

```
torch
torch-geometric
transformers
numpy
pandas
scikit-learn
```

---

## Supervisor

Dr. Arabella Sinclair — University College London, Department of Information Studies

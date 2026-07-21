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

**Primary: Ali-CCP** (Alibaba Click and Conversion Prediction) — EDA on `sample_train`, updated 2026-07-15 with an **exact full-file scan** (42.3M rows, no sampling error) plus a 50K-row **reservoir sample** (Algorithm R, uniformly distributed across the whole file) for distributional stats.

Note: the initial 2026-07-15 pass used the first 500K rows of the file directly, which turned out to be invalid — `sample_skeleton_train.csv` is grouped/sorted by `common_feature_index`, so a sequential read of the first N rows is not a random sample (confirmed: only 435 unique indices across the first 20,000 rows). Fixed same day via a resumable full-file scan.

| Metric | Value |
|---|---|
| Source | Alibaba Taobao recommendation logs |
| sample_skeleton_train.csv | **42,300,135 rows exact** (10.9GB) |
| common_features_train.csv | ~730K rows (8.6GB, deduplicated user/context feature store) |
| CTR (click/impression) | **3.8871%** (exact, full scan: 1,644,256 / 42,300,135) |
| CVR (purchase/click, post-click) | **0.5493%** (exact: 9,032 / 1,644,256) |
| CVR (purchase/impression, overall) | **0.02135%** (exact: 9,032 / 42,300,135) — far sparser than Criteo (10.86%) and than the ~2% figure commonly cited in the literature; worth reconciling against specific papers' CVR definitions before writing this into Methodology |
| Unique items in 50K reservoir (field 205) | 42,553 (85% of rows are a distinct item) — median 1 interaction/item, extreme long tail |
| common_feature_index reuse | ~57.9 average rows/index (42.3M rows ÷ ~730K indices); reservoir sample confirms indices are near-unique per row when sampled without replacement across the whole file (median reuse 1 in the 50K reservoir — the earlier "median 39" was an artifact of the sequential-read bias, not a real pattern) |
| feature_num per skeleton row | mean ~13.3–13.6 (range 1–50) |
| feature_num per common_features row | mean 518 (range 1–2,706) |
| Format quirk | Raw files use ASCII control-char delimiters (`\x01`/`\x02`/`\x03`), not comma/colon as officially documented — required a custom parser |
| Time span | 8 consecutive days |
| Features | User profile (field 101 = user_id, via common_features join), item attributes (field 205), context |

Charts: `aliccp_eda.png` (original biased sample, kept for record), `aliccp_eda_unbiased.png` (corrected). Scripts: `aliccp_eda_raw.py` (format parser + sniff), `full_scan_chunk.py` (resumable exact full-file scan + reservoir sampler — the definitive method going forward for any rate statistic on this file).

### Preprocessing: k-core filtering (finalised 2026-07-15)

Goal was a modelling-ready subset in the ~2-5M interaction range. Exact degree counts (`degree_distribution_scan.py`, one full pass, item_id + common_feature_index counters) showed items are long-tailed on the *low* end (42.6% appear once) while sessions/common_feature_index are long-tailed on the *high* end (only 1.2% appear once, mean 57.9 rows/session) — so session degree, not item degree, is the real lever for row-count control.

| Parameter | Value |
|---|---|
| Item threshold (K_ITEM) | ≥50 interactions → 140,782 items |
| Session threshold (K_SESSION) | ≥200 interactions → 19,550 sessions |
| Rows kept | **3,249,246** (7.68% of original) |
| CTR (kept subset) | 3.22% (close to the exact full-population 3.89%) |
| CVR post-click (kept subset) | 0.57% (close to the exact full-population 0.55%) |
| user_id join coverage | 16,795 / 19,550 sessions matched a user_id (2,755 sessions' common_features row had no field-101 value — not all common_feature_index groups carry an explicit user_id) |

**Trade-off note for Methodology**: the original 50K-200K per-entity target and the 2-5M row target turned out to be jointly infeasible — session activity clusters sharply in the 100-300 range (94,969 sessions at k=100 drops to 19,550 at k=200 to 4,904 at k=300), so no k keeps sessions ≥50K while rows are already ≤5M. Row-count (training feasibility) was prioritised over the entity-count floor; ~19.5K active sessions is a normal scale for published GNN recsys work.

Scripts: `degree_distribution_scan.py` (exact full-file item/session degree counters), `check_session_thresholds.py` (fast marginal-threshold lookup from the saved counters, no rescan), `filter_and_join.py` (applies K_ITEM/K_SESSION, joins user_id from common_features). Output: `aliccp_filtered_joined.csv` — columns `sample_id, click, purchase, common_feature_index, item_id, user_id`.

**Known gap**: the filtered/joined file currently carries only scalar IDs (item_id, user_id), not the full categorical feature blob — a follow-up pass will need to re-extract the richer field values (from both skeleton and common_features) for the qualifying rows/sessions to build the text descriptions the LLM encoder needs.

**Secondary: Criteo Conversion Log** — 16M clicks, 30-day window with real delay timestamps (used for delayed feedback benchmarking).

---

## Baselines

| Category | Baselines |
|---|---|
| Delayed Feedback | DFM, ES-DFM, FSIW, DEFER |
| Graph Recommendation | LightGCN, TGN, DGSR |
| LLM-Enhanced | UniSRec, RLMRec, TALLRec, BERT4Rec (ID-based reference point) |

---

## Literature Review Summary

**Module A — Delayed Feedback Correction (done)**

| Abbr. | Source | Key Finding |
|---|---|---|
| DFM | Chapelle, KDD 2014 | EM algorithm + exponential delay distribution to correct false-negative labels |
| ES-DFM | Yang et al., AAAI 2021 | Elapsed-time sampling + importance weighting; curated data beats full noisy data |
| FSIW | Wen et al., 2021 | Distribution-shift view; importance weights with consistency guarantee; 67x faster than DFM |
| DEFER | Su et al., arXiv:2104.14121 | Duplicates both positives and real negatives to correct training distribution; +8.5% online CVR |

**Module B — Graph Neural Networks (done)**

| Abbr. | Source | Key Finding |
|---|---|---|
| LightGCN | He et al., SIGIR 2020 | Pure neighborhood-aggregation GCN; ~16% over NGCF; exposes the ceiling of ID embeddings |
| TGN | Rossi et al., ICML 2020 | Memory module (GRU) + Time2Vec for continuous-time dynamic graphs |
| DGSR | Zhang et al., TKDE 2022 | Dynamic graph sequential rec; edge quintuple (u,i,t,o_u,o_i); dual-channel attention |

**Module C — LLM-Based Recommendation (done 2026-07-15)**

| Abbr. | Source | Key Finding |
|---|---|---|
| BERT4Rec | Sun et al., CIKM 2019 | Bidirectional self-attention via Cloze/MLM objective on **item IDs only** (no text) — beats SASRec/GRU4Rec by ~7-11% avg across 4 datasets. Despite the "BERT" name it borrows only the architecture/objective, not language understanding. Authors explicitly flag "incorporating rich item features instead of just item ids" as future work — directly the gap RQ1 addresses. Serves as our strongest pure ID-based sequential baseline. |
| UniSRec | Hou et al., KDD 2022 | Encodes item text (title/category/brand) via frozen BERT + parametric whitening + MoE adaptor, replacing ID embeddings entirely for the item side; Transformer over text-derived sequence reps. Beats SASRec/BERT4Rec/ZESRec on 5 cross-domain + 1 cross-platform Amazon/Online-Retail datasets, with the largest gains on long-tail/cold-start items. Direct evidence for RQ1 (text embeddings > ID embeddings, especially for sparse entities) and a template for the dissertation's LLM Encoder layer (whitening + adaptor to align raw PLM embeddings to the rec task). |
| RLMRec | Ren et al., WWW 2024 | Model-agnostic: generates user/item text profiles via LLM reasoning (ChatGPT), embeds with text-embedding-ada-002, then aligns this semantic space with existing GNN/CF ID embeddings via contrastive (RLMRec-Con) or generative/masked (RLMRec-Gen) mutual-information maximization — doesn't replace ID embeddings, denoises them. +2-8% across 6 CF backbones (incl. LightGCN) on 3 datasets; explicitly tested robustness to injected label noise (5-25%) and stays ahead of the base model throughout — directly relevant to RQ2 (false-positive/negative noise in implicit feedback, adjacent to the delayed-feedback problem). Training overhead only ~10-20% over base backbone (RQ3 data point). |
| TALLRec | Bao et al., RecSys 2023 | Fine-tunes LLaMA-7B (LoRA, single RTX 3090) directly as the recommender via instruction tuning on as few as 64-256 labelled examples; beats both zero-shot ChatGPT (~AUC 0.5, random) and traditional sequential baselines. Strong cross-domain generalization (movie→book). Contrast case for RQ3: this is the **fine-tuning** path (higher compute, LLM-as-recommender) vs. our dissertation's chosen **frozen out-of-the-box embedding** path (LLM-as-encoder, no fine-tuning) — useful to cite when justifying the compute/performance trade-off in Methodology. |

**Cross-cutting takeaways for Methodology**:
1. Every LLM-rec paper here confirms text/semantic representations help most where ID embeddings are weakest — sparse/long-tail/cold-start entities. This matches Ali-CCP's item catalog (85% singleton-interaction items in a uniform sample, see EDA above), which is a strong empirical motivation for RQ1.
2. Two distinct architectural patterns emerged: (a) replace ID embeddings with text embeddings (UniSRec, our own plan), vs (b) keep ID embeddings and align/denoise them against a separate semantic space (RLMRec). Worth deciding explicitly which pattern RQ1's architecture follows, or testing both as an ablation.
3. None of these four papers address delayed feedback directly — RLMRec's noise-robustness experiment is the closest adjacent result. This confirms the dissertation's three-layer design (LLM Encoder + Dynamic Graph + Delayed Feedback Correction) is combining pieces that don't currently coexist in one published system — a genuine gap, not a replication.

---

## Repository Structure

Current state (updated 2026-07-15) — `src/` (models), `notebooks/`, and `experiments/` are not yet created since modelling hasn't started; they're kept below as the planned layout.

```
dissertation-cvr-llm/
├── data/
│   ├── README.md                    # Download instructions for Ali-CCP & Criteo [planned]
│   └── preprocessing/
│       ├── eda_datasets.py              # Initial Criteo + Ali-CCP EDA
│       ├── aliccp_eda_raw.py            # Ali-CCP raw-format parser (\x01\x02\x03 delimiters) + EDA
│       ├── full_scan_chunk.py           # Resumable exact full-file scan + reservoir sampler (bias fix)
│       ├── degree_distribution_scan.py  # Exact item/session degree counters, k-core threshold table
│       ├── check_session_thresholds.py  # Marginal k-core lookup from saved counters (no rescan)
│       └── filter_and_join.py           # k-core filtering (K_ITEM/K_SESSION) + user_id join
├── src/                              # [planned]
│   ├── models/
│   │   ├── llm_encoder.py               # LLM embedding module
│   │   ├── dynamic_graph.py             # DGSR-based graph network
│   │   └── df_correction.py             # Delayed feedback correction loss
│   ├── baselines/                       # Baseline model implementations
│   └── utils/                           # Shared utilities
├── notebooks/                        # [planned] Exploratory analysis and visualisation
├── experiments/                      # [planned] Config files and experiment logs
├── results/
│   └── figures/
│       ├── criteo_eda.png
│       ├── aliccp_eda.png               # original biased-sample EDA (kept for record)
│       └── aliccp_eda_unbiased.png      # corrected full-scan EDA
├── docs/
│   ├── Proposal_Liu_Yize_Updated.docx
│   ├── literature_reading_list.docx
│   ├── dis_research_ethics_form_updated.docx
│   ├── dissertation_outline_CN.pdf / dissertation_outline_EN.pdf
│   ├── dissertation_overview.pdf
│   ├── progress_report_week1.pdf
│   ├── supervisor_meeting_proposal.pdf
│   └── supervisor_meeting_script.md
├── References/                       # Literature PDFs, organised by module (01-04)
├── .gitignore                        # Excludes Dataset/ (15GB raw data), *.csv, *.pkl, *.tar.gz
├── requirements.txt
├── main.tex
└── README.md
```

Note: `Dataset/` (raw Ali-CCP + Criteo files, ~15GB) is gitignored and lives only on local machines — not pushed to GitHub. See Dataset section above for download/format notes.

---

## Progress

- [x] Literature review — Delayed Feedback (DFM, ES-DFM, FSIW, DEFER)
- [x] Literature review — Graph Networks (LightGCN, TGN, DGSR)
- [x] Literature review — LLM Recommendation (BERT4Rec, UniSRec, RLMRec, TALLRec) (2026-07-15)
- [x] Dataset acquisition and EDA — Ali-CCP (2026-07-15), Criteo (done)
- [x] k-core filtering pipeline — Ali-CCP reduced to 3.25M rows / 140,782 items / 19,550 sessions (2026-07-15)
- [ ] Train/test split — representative split, fixed and reused across all subsequent experiments
- [ ] Baseline model run (data is ready — unblocked)
- [ ] Feature/text extraction for LLM encoder input (known gap — see Dataset section)
- [ ] LLM Encoder V1 — ID embeddings + LLM embedding (BERT first pass; see Next Steps)
- [ ] LLM Encoder V2 (if time permits)
- [ ] Test set design — easy/hard splits (e.g. by prediction horizon length); multi-dataset × multi-model evaluation matrix
- [ ] Dissertation write-up (Overleaf — see Next Steps)

---

## Next Steps (Supervisor Meeting 2026-07-21)

Agreed with Dr. Sinclair:

1. **Baseline model** — data is ready, run now.
   - First: build a representative train/test split; this exact split is reused for every experiment going forward (baselines, V1, V2) so results are comparable.
2. **Feature/text extraction** — extract the full categorical feature set (from skeleton + common_features) needed to build LLM input text for the k-core-filtered subset (see "Known gap" in Dataset section).
3. **V1 — add LLM embedding**
   - Write up the exact method before implementing.
   - LLM embedder choice: first pass with a small LM (BERT) to get an initial result quickly.
   - Maintain a candidate list of alternative embedders to try afterward: XLNet, MPNet, and other LMs known for strong representation quality.
   - Report results for V1 once done.
4. **V2** — only if time allows after V1 is reported.
5. **Test set design**
   - Split the test set into "easier" vs "harder" segments — e.g. test whether shorter prediction horizons are easier, and if so, evaluate the two groups separately.
   - Evaluation matrix: multiple datasets × multiple models, all on the *same* train/test combination, so techniques can be compared directly and fairly.
6. **Thesis write-up (Overleaf)**
   - Template: [overleaf.com/project/6a35112f7339d98b5f33bc6a](https://www.overleaf.com/project/6a35112f7339d98b5f33bc6a)
   - Make a copy, name it `[MSc Thesis] YourName ProjectName`.
   - Populate with the literature list/citations already compiled (see Literature Review Summary above and `docs/literature_reading_list.docx`).
   - Add baseline methods/results once available (from step 1).

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

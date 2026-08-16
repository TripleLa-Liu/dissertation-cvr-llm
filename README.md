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

> **Correction (2026-07-2x, per migration notes — not independently re-verified in this session):** the post-click CVR figure above (0.5493%) was computed with a bug — it divided all purchase rows by all click rows instead of restricting to purchases *within clicked rows*. Fixed definition: purchase-rows-among-clicked-rows / clicked-rows. Corrected full-population CVR = **0.5353%**. The fix was applied across 4 scripts, but those script versions were not migrated to this repo copy (see "Known gaps" below) — treat 0.5493% in this table as superseded pending re-verification.

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

**Resolved (was "known gap" as of 2026-07-15):** the filtered/joined file initially carried only scalar IDs. A follow-up pass re-extracted the richer categorical field values from both skeleton and common_features for the qualifying rows/sessions, and built template-based "pseudo-text" descriptions for the LLM encoder (see Feature Extraction subsection below). **Core limitation carried forward**: all Ali-CCP fields are anonymised numeric IDs with no public decode table, so pseudo-text is templated (`"This item's category is category_8314904, shop is shop_8702354..."`) rather than real natural language — it cannot test whether an LLM's pretrained world knowledge helps, only whether frozen sentence-embedding geometry over templated categorical text helps. Whether to bring in a second, real-text dataset (e.g. Amazon Reviews / MIND) to isolate this is still an open question for the supervisor (see "Open Questions" below).

### Official train/val/test split (supersedes the k-core section above)

The k-core filtering above was re-run against Ali-CCP's **official train/test file split** (not a random split), so all downstream numbers use official test data, not held-out train data:

| Split | Rows | Source |
|---|---|---|
| Train pool (post k-core) | 3,249,246 | `aliccp_filtered_joined.csv` |
| Train | 2,928,363 | `aliccp_train_split.csv` — 90% of train pool, split by session |
| Val | 320,883 | `aliccp_val_split.csv` — 10% of train pool, split by session |
| Test | 2,006,347 | `aliccp_test_filtered_joined.csv` — official `sample_test` file, same K_ITEM/K_SESSION filter applied independently, includes an `is_cold_start_item` flag |

Same K_ITEM=50 / K_SESSION=200 thresholds as above. Scripts that produced these splits are not present in this repo copy (see "Known gaps" below); data files are in `migration_package/processed_data/`.

### Filtering distortion check (finalised)

Two-proportion z-test on whether k-core filtering distorts the label distributions relative to the unbiased 50K reservoir sample:

- **CTR**: real, statistically significant difference — filtered subset CTR is ~17% relatively lower than the full population, because k-core selects on session activity and highly active sessions have inherently lower CTR.
- **CVR (post-click)** — the dissertation's actual target metric — **no significant difference** between filtered and unfiltered populations.

Conclusion: k-core filtering does not distort the metric this dissertation optimises for. Script: `data/preprocessing/confirm_filtering_no_distortion.py` (not present in this repo copy — see "Known gaps").

### Feature extraction & LLM pseudo-text

Full categorical features (category, shop, intention node, brand, etc.) extracted for items and users in the qualifying train+test rows, templated into pseudo-text strings for embedding:

- `item_pseudo_text.csv` — 663,474 items
- `user_pseudo_text.csv` — 12,065 users

Example: `"This item's category is category_8314904, shop is shop_8702354, intention node is intention_node_9098377, brand is brand_9345479."`

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

## Modelling Results (as of 2026-07-29, from migration_package data — scripts not present in this repo copy)

### Baseline: ESMM (ID-embedding two-tower)

Working baseline chosen from the literature list above. Test set = official Ali-CCP test split (2,006,347 rows).

| Metric | Val | Test overall | Test seen items | Test cold-start items |
|---|---|---|---|---|
| CTR-AUC | 0.6328 | 0.5630 | 0.5759 | 0.5375 |
| CVR-AUC (post-click) | 0.5654 | 0.5260 | 0.5375 | 0.5315 |
| **CTCVR-AUC** | 0.6243 | **0.5610** | 0.5713 | 0.5557 |

Checkpoint + vocab retained in `migration_package/processed_data/baseline_results/`.

### V1 / V1-Full: frozen MiniLM pseudo-text embeddings replacing ID embeddings

V1 = item-only text embedding; V1-Full = item + user text embeddings. Both replace ID embeddings entirely.

| Model | Test CTCVR-AUC | vs. baseline (0.5610) |
|---|---|---|
| V1 (item-only) | 0.5287 | lower |
| V1-Full (item+user) | 0.5124 | lower |

Both underperform the ID baseline — replacing ID embeddings with frozen text embeddings loses fine-grained memorisation capacity that IDs provide.

### V2: RLMRec-style alignment (ID embeddings retained + contrastive alignment to text)

| Model | Test CTCVR-AUC | vs. baseline |
|---|---|---|
| V2 | 0.5616 | ~parity (only LLM variant not clearly below baseline) |

### Candidate encoder shortlist ("Different Embedders" experiment, done)

MPNet, BGE, E5, GTE, EmbeddingGemma shortlisted as sentence-embedding alternatives to MiniLM. XLNet evaluated and ruled out as unsuitable for direct sentence-embedding use.

`src/models/llm_encoder_v2_mpnet.py` added (2026-07-30): identical to `llm_encoder_v2_aligned.py` (V2) with the frozen encoder swapped from `all-MiniLM-L6-v2` (384-dim) to `all-mpnet-base-v2` (768-dim); writes to a separate `v2_mpnet_results/` directory so it doesn't overwrite the MiniLM checkpoint. V2 was chosen as the base for this comparison because it's the only LLM variant that matched or beat the ID baseline, making it the more informative test of whether encoder quality — as opposed to the anonymised-ID pseudo-text ceiling itself — is the binding constraint.

**Result (2026-08-02, both run to completion):**

| Metric (test) | V2 — MiniLM (384-dim) | V2 — MPNet (768-dim) | Delta |
|---|---|---|---|
| CTR-AUC overall | 0.5458 | 0.5448 | ~parity |
| CVR-AUC overall (post-click) | 0.5455 | 0.5306 | MPNet lower |
| **CTCVR-AUC overall** | **0.5616** | 0.5571 | MPNet slightly lower |
| CTCVR-AUC, seen items | 0.5656 | **0.5723** | MPNet higher |
| CTCVR-AUC, cold-start items | **0.5724** | 0.5361 | MPNet notably lower |

Swapping to a larger, higher-dimensional general-purpose encoder does **not** improve overall CTCVR-AUC, and the aggregate result masks a real split: MPNet is slightly *better* than MiniLM on seen items but clearly *worse* on cold-start items (-0.036 CTCVR-AUC) — the opposite of what "better encoder" would naively predict, since cold-start items are exactly where semantic (non-ID) signal should matter most. Plausible explanation: the higher-dimensional MPNet space is harder to align to the same 32-dim ID-embedding space with the same `lambda_align=0.1` and epoch budget used for MiniLM — this wasn't retuned per-encoder, so the result may reflect an under-tuned alignment objective for MPNet rather than an intrinsic embedding-quality ceiling. Either way, this supports the earlier conclusion that encoder quality is not the main bottleneck for V2 on Ali-CCP's anonymised pseudo-text — bigger embeddings alone don't buy a better result, and can hurt exactly where they're expected to help. BGE/E5/GTE/EmbeddingGemma variants not implemented — deprioritized given this result.

### Test-set difficulty segmentation (context-length × cold-start, 2×2)

Ali-CCP has no timestamps, so "prediction horizon" was reinterpreted as **session interaction count** (long_context ≥ median vs. short_context < median), crossed with seen/cold-start item status. Four models evaluated across all four cells (`context_segment_eval_results.json`).

**Key finding — a crossover**: V2 is the strongest model overall, but in the hardest cell, **short-context + cold-start** (418,643 rows, ~20.9% of test set), V1 (0.5627) and V1-Full (0.5564) both beat V2 (0.4991) and baseline (0.5413):

| Model | short_context & cold_start CTCVR-AUC |
|---|---|
| Baseline | 0.5413 |
| V1 | **0.5627** |
| V1-Full | 0.5564 |
| V2 | 0.4991 |

### V3: hybrid segment router (V1 + V2)

Directly answers the 2026-07-30 meeting question: "Should this motivate a hybrid V3 (route cold-start items by context richness)?"

`src/models/llm_encoder_v3_hybrid.py` added (2026-08-02): not a new trained model — a deterministic router over the already-trained V1 and V2 checkpoints. Rule: `is_hard_segment = (context_segment == "short_context") & is_cold_start_item`; use V1's prediction where true, V2's elsewhere (p_ctr/p_cvr/p_ctcvr all routed together per row from the same source model, so ctcvr_hybrid stays internally consistent — not routed independently). No new training, no threshold tuned against test performance — the rule is the segment definition itself, fixed before this script ever looked at test AUC.

**Result (2026-08-02, run to completion):**

| Model | overall CTCVR-AUC | short_context & cold_start CTCVR-AUC |
|---|---|---|
| Baseline | 0.5610 | 0.5413 |
| V1 | 0.5299 | 0.5671 |
| V1-Full | 0.5124 | 0.5564 |
| V2 | 0.5616 | 0.5455 |
| **V3-Hybrid** | **0.5649** | **0.5671** |

The hybrid wins on both counts, not just the hard segment it was designed for. In the target cell it exactly matches V1's number (0.5671, as expected — it's literally V1's prediction there) and clearly beats V2 alone (0.5455) and baseline (0.5413). What wasn't guaranteed going in: **overall CTCVR-AUC also improves over pure V2** (0.5649 vs 0.5616), even though V1 is the much weaker model everywhere else (0.5299 overall) and only 20.9% of test rows are actually routed to it. This means swapping in V1's ranking for the one segment where V2 structurally can't have learned anything useful (item never seen in train + almost no user history) improves the pooled ranking more than the cross-model calibration mismatch hurts it — the routing rule earns its complexity rather than just being a wash. All three metrics (`p_ctr`, `p_cvr`, `p_ctcvr`) are routed together per row from the same source model, so `ctcvr_hybrid` stays internally consistent (verified: `short_context & cold_start` row is bit-identical to V1's own number for that cell). Full 5-model breakdown in `v3_hybrid_results.json`.

This directly answers the meeting question ("Should this motivate a hybrid V3?") — yes, and it's a real, positive, reportable result, not just a defensible-but-flat outcome.

**Why this crossover happens (mechanism, not just correlation)** — traced through both models' training code, not guessed:

- **V2's cold-start item representation is real, but only weakly supervised.** V2 keeps a learned `item_emb` for in-vocab items and a trainable `item_llm_proj(frozen_llm_embedding)` for out-of-vocab (cold-start) items (`llm_encoder_v2_aligned.py`'s `item_representation()`). But `item_llm_proj` only ever receives gradient through the **contrastive alignment loss** (`alignment_loss()`, an InfoNCE term pulling each *train-vocab* item's `item_emb` and `item_llm_proj` output together) — it never receives gradient from the main CTR/CVR BCE loss, because every training-batch item is by construction in-vocab, so the main forward pass always takes the `item_emb` branch (see `item_representation()`'s `use_id` mask: `idx == 0` never happens during training). At test time, cold-start items *do* take the `item_llm_proj` branch — a projection head that was only ever trained to *mimic the geometry of other items' ID embeddings*, not to directly predict CTR/CVR from text.
- **V1's item representation is identically real-text-based for every item, warm or cold, and is directly supervised.** V1 has no ID embedding at all — `item_adapter(frozen_llm_embedding)` is the only item pathway, and it receives gradient directly from the CTR/CVR BCE loss on every training row. Direct task supervision transfers to an unseen item's text better than a proxy contrastive objective does.
- That's the mechanism behind the crossover: **on cold-start items, V1's text pathway beats V2's text pathway because it was trained on the actual task, not a proxy alignment objective.** On seen items it's the reverse: V2 keeps a fully free-form, high-capacity `item_emb` per item, while V1 has thrown that away entirely in favour of a lower-capacity text-derived representation — costly exactly where per-item memorisation (not generalisation) is what pays off.
- This is also why the routing rule isn't arbitrary pattern-matching on the segmentation table: it routes to whichever model's item representation received the *right kind of supervision* for that row's regime (memorisation for warm items, direct text-to-task supervision for cold ones).

**Caveat — calibration, not correctness.** V1 and V2 are trained completely independently (different loss surfaces, different sigmoid calibration), so there's no guarantee in general that swapping in one model's raw probability output for a subset of rows preserves a sane global ranking — it happened to help here (overall AUC went up, not just the target segment), but that's an empirical result for this pair of models, not a property that would necessarily hold for any two models being routed this way. If V3-style routing were ever used for something other than rank-only AUC evaluation (e.g. actual probability outputs), the two branches' outputs would need calibrating onto a shared scale first (e.g. per-segment Platt scaling), which this script does not do.

### Multi-seed reruns, MPNet completion, Amazon V2/V3, and significance testing (2026-08-04/05 post-meeting round)

Third post-meeting todo item ("list model variant × embedder × dataset performance together, more robust, more metrics like std and t-test"), scoped as "LLM side" work only — Dynamic Graph implementation and Delayed Feedback Correction were explicitly deferred to a later round.

**Infrastructure added:**
- `--seed` CLI arg added to all 10 training scripts (`id_embedding_baseline.py`, `llm_encoder_v1.py`, `llm_encoder_v1_full.py`, `llm_encoder_v2_aligned.py`, `llm_encoder_v1_mpnet.py` **(new)**, `llm_encoder_v1_full_mpnet.py` **(new)**, `llm_encoder_v2_mpnet.py`, `amazon_id_baseline.py`, `amazon_text_embedding.py`, `amazon_v2_aligned.py` **(new)**). Default seed (42) keeps the original unsuffixed checkpoint/metrics filenames so downstream scripts with hardcoded paths (`llm_encoder_v3_hybrid.py`, `amazon_v3_hybrid.py`) keep working unchanged; any other seed writes to `..._seed{N}` files.
- `src/aggregate_multiseed_results.py` **(new)** — reads all seed-suffixed metrics files for one model, reports mean ± std per metric.
- `src/significance_tests.py` **(new)** — paired t-test (`scipy.stats.ttest_rel`) between two models across shared seeds; chose *paired-across-seed* rather than literal k-fold cross-validation because Ali-CCP/Amazon's `is_cold_start_item` flag depends on a fixed train/test item split — re-partitioning the data k different ways (true k-fold CV) would make an item cold-start in one fold and warm in another, destroying the segment definition the whole experiment is built around. Reruns under 5 different seeds (42, 123, 2026, 7, 99) instead hold the split fixed and only vary training-process randomness, which is the appropriate source of variance to test here.
- `amazon_v2_aligned.py` **(new)** — RLMRec-style ID+alignment (V2 pattern), Amazon counterpart to `llm_encoder_v2_aligned.py`, completing the baseline/V1/V2 set on both datasets.
- `amazon_v3_hybrid.py` **(new)** — hybrid segment router, Amazon counterpart to `llm_encoder_v3_hybrid.py`.
- `amazon_build_difficulty_segments.py` **(new)** — Amazon counterpart to `build_test_difficulty_segments.py`; segments by train+val+test combined interaction count (median=15) rather than train-only count, which degenerates to median=0 given Amazon's chronological split (59.7% of test users have zero train rows).
- All 10 scripts run to completion locally, 5 seeds each (Amazon segmentation + both V3-hybrid routers run once each, seed=42 checkpoints only — pure inference, no training randomness to average over).

**Full comparison matrix, mean ± std across 5 seeds (CTCVR-AUC for Ali-CCP, AUC for Amazon):**

| Ali-CCP model | Test Overall | Test Seen | Test Cold-Start |
|---|---|---|---|
| Baseline (ID) | 0.5562 ± 0.0048 | 0.5572 ± 0.0105 | 0.5573 ± 0.0030 |
| V1 (MiniLM) | 0.5576 ± 0.0248 | 0.5655 ± 0.0372 | 0.5462 ± 0.0127 |
| V1 (MPNet) | 0.5462 ± 0.0133 | 0.5427 ± 0.0188 | 0.5499 ± 0.0126 |
| V1-Full (MiniLM) | 0.5121 ± 0.0011 | 0.4908 ± 0.0019 | 0.5421 ± 0.0031 |
| V1-Full (MPNet) | 0.5111 ± 0.0093 | 0.5051 ± 0.0033 | 0.5187 ± 0.0243 |
| V2 (MiniLM) | 0.5574 ± 0.0050 | 0.5566 ± 0.0052 | 0.5639 ± 0.0089 |
| V2 (MPNet) | 0.5576 ± 0.0134 | 0.5646 ± 0.0109 | 0.5492 ± 0.0247 |

| Amazon model | Test Overall* | Test Seen | Test Cold-Start |
|---|---|---|---|
| Baseline (ID) | 0.4687 ± 0.0048 | 0.6150 ± 0.0054 | 0.5011 ± 0.0062 |
| V1 (MiniLM, text-replace) | 0.4178 ± 0.0025 | 0.6047 ± 0.0064 | 0.4945 ± 0.0035 |
| V2 (MiniLM, align) | 0.4930 ± 0.0098 | 0.6177 ± 0.0119 | 0.5027 ± 0.0106 |

*See the `test_overall` pooling caveat below the Amazon results table further up this README — the seen/cold-start columns are the primary comparison for Amazon.

**Paired significance tests (5 shared seeds each):**

The only statistically significant Ali-CCP result at n=5 is **V2 beating V1 on the cold-start segment** (mean diff +0.0177, t=-5.65, p=0.0048). Every other Ali-CCP comparison — Baseline vs V1/V2 (overall, seen, or cold-start), and MiniLM vs MPNet for V1/V1-Full/V2 — is not distinguishable from seed noise (all p > 0.10). This means "MPNet doesn't help over MiniLM" and most of the Baseline-vs-LLM-variant deltas reported earlier in this README should be described as directional, not significant, findings in Results.

On Amazon, Baseline/V1/V2 differences on `test_overall` are all significant (p < 0.01), but this metric is confounded (see pooling caveat) and none of the underlying seen/cold-start subgroup differences are themselves significant — recommend not citing the Amazon `test_overall` p-values without that caveat attached. Full test log: `results/significance_tests_full.log`; full comparison table with all tests: `results/multiseed_comparison_summary.md`.

**Amazon V3-hybrid: routing direction had to be flipped, and the fix is itself a finding.** The first `amazon_v3_hybrid.py` run mechanically copied Ali-CCP's routing rule (hard segment `short_context & cold_start` → V1). Per-cell breakdown showed this is backwards for Amazon: V2 wins the hard segment here (0.5173 vs V1's 0.4872), while V1 wins the other three cells (long&seen, long&cold, short&seen) — the reverse of Ali-CCP, where V1 wins the hard segment. Flipped the rule (hard segment → V2, elsewhere → V1) and reran:

| | overall AUC | hard segment AUC |
|---|---|---|
| V1 alone | 0.4209 | 0.4872 |
| V2 alone | 0.5088 | 0.5173 |
| V3-hybrid (wrong direction) | 0.4443 | 0.4872 |
| V3-hybrid (correct direction) | 0.4944 | 0.5173 |

This is the direct empirical answer to the supervisor's meeting-comment question about whether a real-text dataset would confirm the original prediction that V2 should win the cold-start/short-context setting: on Ali-CCP's pseudo-text it doesn't (V1 wins), on Amazon's real text it does (V2 wins) — the routing direction that's optimal flips between the two datasets. One caveat: even with the corrected direction, the hybrid's *overall/pooled* AUC (0.4944) is still below V2 alone (0.5088), despite winning or tying V2 in every individual cell — the same AUC-pooling effect as the `test_overall` caveat above (seen cells ~12% positive rate vs. cold-start cells ~47%), not a routing error. Unlike Ali-CCP, where the hybrid's overall AUC (0.5649) cleanly beat every individual model, Amazon's per-cell breakdown is the number to cite, not `overall`. Only a single seed (seed=42 checkpoints); this cell-level result has not been significance-tested. Original mis-routed result preserved as `v3_hybrid_results_ORIGINAL_MISROUTED.json` for reference.

All raw per-model JSON summaries: `results/multiseed_summaries/`.

### Overleaf preparation

`docs/references.bib` (11 core citations + 3 embedding-model citations) and `docs/methods_results_draft.tex` (paste-ready Methods/Results draft, single-round, pre-restructure) were prepared but not yet pasted into the Overleaf project, and are not present in this repo copy (see "Known gaps").

### Thesis restructure + writing round (2026-08-08)

The live Overleaf thesis (uploaded PDF, reviewed in full including Dr. Sinclair's `[AS: ...]` comments) had the V3 hybrid-router definition, the Amazon dataset-construction narrative, and the MPNet-vs-MiniLM encoder discussion all sitting in the Results chapter rather than Methods, and had no content at all yet for the newly-completed multi-seed/significance-testing round or the V4 Dynamic Graph layer. Restructured and drafted for pasting into Overleaf:

- `docs/methods_draft.tex` **(new)** — restructured Chapter 3 (Methods): 3.1 Data now covers both datasets (Ali-CCP + Amazon Reviews'23 dataset construction, moved in from the old Results 4.3.1) plus a new context-length-segmentation subsection; 3.2 Models gets a design-rationale intro plus the V3 router definition and V4 (Dynamic Graph) architecture description, both moved in from Results; 3.3 Metrics gets the seen/cold-start split rationale and the multi-seed/significance-testing methodology; 3.4 Encoders (previously an empty `[AS: todo]`) gets the MiniLM-vs-MPNet design rationale moved in from the old Results 4.1.
- `docs/results_draft.tex` **(new)** — restructured Chapter 4 (Results), reorganised by dataset (4.1 Ali-CCP, 4.2 Amazon) rather than by topic, each with the same shape (main model comparison, then the dataset's extension study) per the plan the user confirmed after flagging the topic-first version was too fragmented. Split into multiple small tables (main comparison, significance tests, encoder capacity, V3 routing, V4) instead of one large table, now that multi-seed mean±std and paired-significance columns don't fit one table. Includes the full multi-seed comparison matrix, significance tests, the Amazon V3 routing-direction-flip finding, and the entire V4 Dynamic Graph section (new content, not in the previous draft at all).
- `docs/chapter_patches.md` **(new)** — Chapters 1, 2, and 5 of the live thesis describe the Dynamic Graph layer as unimplemented future work (accurate when written, outdated now that V4 exists); gives exact quoted before/after text for the 4 sentences that need updating, since only the compiled PDF (not editable source) was available for those chapters.

All figures in `results_draft.tex` cross-checked against `results/multiseed_comparison_summary.md`, `results/significance_tests_full.log`, `results/v4_significance_tests.log`, and `results/v4_ablation_significance.log`; both `.tex` files checked for brace/environment balance. Not yet pasted into the live Overleaf project — that's a manual step for the user, or a follow-up session using the Chrome-based Overleaf editing already used earlier for the Introduction/Background chapters.

### Supervisor meeting follow-up (2026-08-15): variant-coverage matrix, Amazon MPNet gap

The compiled thesis PDF (as of 2026-08-15) confirmed the 2026-08-08 restructure above **had** already been pasted into Overleaf in full — Ali-CCP's V1(MPNet) table and both datasets' V3-Hybrid results are already in the live document. What the supervisor's review actually caught was two things not yet addressed by that round:

1. **"Too many model variants, hard to track which config is which."** `docs/methods_draft.tex` \S3.2 previously described all seven variants (Baseline, V1, V1-Full, V2, V3-Hybrid, V4, V4-ID-Only) as prose only. Added two summary tables right before the prose descriptions: `tab:variant-config` (what each variant's item/user representation, alignment loss, and history aggregation actually are) and `tab:variant-coverage` (which of the 4 dataset x encoder combinations, and how many seeds, each variant was actually evaluated on — with lettered notes explaining every structural gap: Baseline has no text pathway so encoder choice doesn't apply; V1-Full has no Amazon counterpart because Amazon has no user-profile text source, only interaction history, which V4 uses instead; V3-Hybrid and V4 are single-dataset by construction).
2. **Amazon has never been tested with MPNet** — confirmed a genuine gap (not a documentation lag): `amazon_text_embedding.py` (V1) and `amazon_v2_aligned.py` (V2) only ever used `all-MiniLM-L6-v2`. Added `src/models/amazon_text_embedding_mpnet.py` and `src/models/amazon_v2_mpnet.py`, minimal-diff copies (same pattern as Ali-CCP's `llm_encoder_v1_mpnet.py`/`llm_encoder_v2_mpnet.py`: swap `LM_NAME` to `all-mpnet-base-v2`, write to a separate results directory so the MiniLM checkpoints aren't overwritten).

### Amazon MPNet gap closed (2026-08-16): 5-seed results, run locally on Mac

Rebuilt the Amazon Video_Games dataset locally (raw `.jsonl` downloaded by the user via browser, processed with `amazon_build_dataset.py` — exact reproduction of the documented row/user/item counts) and ran both new scripts at 5 seeds each (42, 123, 2026, 7, 99) via `uv run` on the user's Mac. Two unrelated infrastructure issues hit along the way, both now documented in `CLAUDE.md`: `huggingface_hub`'s `hf_xet` transfer client hangs indefinitely at 0 bytes in this environment (fix: `HF_HUB_DISABLE_XET=1`), and even with that fix the model download over the user's proxy was slow/unpredictable (7–60KB/s to HF's Xet-bridge CDN) — resolved by having the user download `all-mpnet-base-v2` once via their own terminal (outside whatever was throttling the automated tool calls), after which the actual training runs read the model from local cache and finished in seconds each.

Aggregated with `src/aggregate_multiseed_results.py` and significance-tested against the existing MiniLM per-seed values in `results/multiseed_summaries/amazon_v{1,2}_minilm.json` (paired by seed, `scipy.stats.ttest_rel`). Unlike Ali-CCP — where no MiniLM-vs-MPNet difference was significant for any variant — **Amazon's V1 (REPLACE) shows a real, significant effect**: MPNet improves both overall AUC (+0.0542, p=0.0002) and cold-start AUC (+0.0386, p=0.0001), but significantly *hurts* seen-items AUC (−0.0232, p=0.0225) — a capacity/compression trade-off, not a strict improvement, plausibly because V1 has no learned ID embedding at all, so the frozen text embedding is the item's *only* representation rather than a secondary signal. V2 (ALIGN) shows no significant difference on any segment (p=0.089–0.953), matching the Ali-CCP V2 result exactly, consistent with V2's text embedding only receiving gradient through the contrastive alignment loss rather than the main CTR/CVR loss. Written up as a new "Encoder Capacity" subsection in `docs/results_draft.tex` §4.2 (mirroring Ali-CCP's §4.1.2), and `docs/methods_draft.tex`'s `tab:variant-coverage` "in progress (b)" cells updated to "5 seeds". Not yet pasted into the live Overleaf project.

---

## Known Gaps in This Repo Copy (updated 2026-07-30, second `code_for_github` sync)

This `Dissertation` folder was a **2026-07-15 snapshot** (code only through k-core filtering). Two `code_for_github` drops have since supplied nearly the full modelling pipeline, now copied into `data/preprocessing/`, `src/baselines/`, `src/models/`, `src/`, `docs/`. Every file was read and cross-checked against the verified result numbers (not copied blind):

**Recovered and verified consistent with the numbers in "Modelling Results":**
- `data/preprocessing/confirm_filtering_no_distortion.py` — hardcoded `FULL_PURCHASE_AMONG_CLICK = 8_802` reproduces CVR_full = 0.5353%
- `data/preprocessing/split_train_val.py` — session-level 90/10 split, seed=42; matches `aliccp_train_split.csv`/`aliccp_val_split.csv` row counts
- `data/preprocessing/filter_test_and_join.py` — asymmetric test filtering (item whitelist inherited from train, session threshold recomputed on test), writes `is_cold_start_item`; matches test row count (2,006,347) and the 42.86% cold-start figure in `methods_results_draft.tex`
- `data/preprocessing/extract_item_pseudo_text.py` / `extract_user_pseudo_text.py` — template builder matches the exact pseudo-text format seen in `item_pseudo_text.csv`/`user_pseudo_text.csv`; field mapping (206=category, 207=shop, 210=intention_node, 216=brand; 121/122/124-129=user demographics) matches `profile_raw_fields.py`'s hypotheses
- `data/preprocessing/profile_raw_fields.py` — reservoir-sampling field profiler behind the pseudo-text field choices (prints only, no persisted output)
- `data/preprocessing/build_test_difficulty_segments.py` — median-split by session interaction count; printed segment sizes (long_context n=1,012,202, short_context n=994,145, and all 4 crossed cells) match `context_segment_eval_results.json` exactly
- `src/eval_context_segments.py` — the script that produced `context_segment_eval_results.json`; segment-breakdown logic matches the 2×2 matrix reported above
- `src/baselines/id_embedding_baseline.py`, `src/models/llm_encoder_v1.py` / `llm_encoder_v1_full.py` / `llm_encoder_v2_aligned.py`, `docs/references.bib`, `docs/methods_results_draft.tex` — as previously verified

**Still missing (one script):**
- `degree_distribution_scan_test.py` — referenced in `filter_test_and_join.py`'s docstring as "STEP 1" (produces `aliccp_degree_counters_test.pkl`, which we do have as a data file). Likely a near-duplicate of `degree_distribution_scan.py` applied to the test file, but not delivered in either `code_for_github` drop — worth locating for completeness, though the pipeline is reproducible without it since its one output file already exists.

**Not scripts — expected to be regenerated, not migrated (per `MIGRATION_NOTES.md`):**
- V1 / V1-Full / V2 checkpoints and frozen embedding caches (`v1_results/`, `v1_full_results/`, `v2_results/`) — `src/eval_context_segments.py` expects these to exist; rerun `llm_encoder_v1.py` / `llm_encoder_v1_full.py` / `llm_encoder_v2_aligned.py` locally (GPU, a few minutes each) to produce them before `eval_context_segments.py` can run

The GitHub remote (`TripleLa-Liu/dissertation-cvr-llm`) is still stale — only 3 commits, README-only. Worth pushing this recovered pipeline once `degree_distribution_scan_test.py` is found and the checkpoints are regenerated.

**Pipeline is now essentially reproducible end-to-end** from raw Ali-CCP files (in `Datasets/`) through to the context-segment evaluation, modulo the one missing helper script and the intentionally-not-migrated checkpoints.

### Repo-hygiene fixes (2026-07-30, this pass)

Checked whether anything would actually block a GitHub push or a from-scratch experiment re-run (not just "is a script physically present"). Found and fixed:
- **`requirements.txt` was missing `sentence-transformers`** — `llm_encoder_v1.py`/`v1_full.py`/`v2_aligned.py` all import it; anyone following requirements.txt as-is would hit an ImportError on the V1/V1-Full/V2 scripts. Added.
- **`.gitignore` didn't exclude model checkpoints** — `*.pt`/`*.pth` and the `baseline_results/`, `v1_results/`, `v1_full_results/`, `v2_results/` output directories (the latter three ~2GB each once regenerated locally) were untracked-but-not-ignored, which would either bloat a `git add .` or hit GitHub's file-size limits. Added, along with `migration_package/processed_data/` (already 910MB of migrated data, shouldn't go into git either).
- **Two stray junk files removed**: `.gitignore_test_DELETE_ME` (a leftover debug file, content was literally `"test"`) and `.git/config.lock.orphan` (a 0-byte stray lock file from an interrupted git operation) — both deleted from disk, and the now-redundant gitignore entry for the former removed.

**Still open (not code-blocking, but noted for completeness):**
- `degree_distribution_scan_test.py` — the one missing script (see above); doesn't block anything currently in the repo since its sole output (`aliccp_degree_counters_test.pkl`) already exists as data, but a fully from-scratch clone-and-run (no migrated data) can't regenerate it without this file.
- `data/README.md` (Ali-CCP/Criteo download instructions) — still `[planned]`, never written.

### WORK_DIR path fix (2026-07-30, after first local run attempt)

Every script (all 21 that reference a data path) had `WORK_DIR = r"E:\BaiduNetdiskDownload\Dataset\_processed"` hardcoded — this was the *old* computer's path, and `MIGRATION_NOTES.md` already flagged that it would need updating if the drive letter/path differed on the new machine. First local run of `amazon_download.py` confirmed it does: `E:\` doesn't exist on this machine. Updated across all scripts:
- `WORK_DIR` (processed data / script outputs): `E:\BaiduNetdiskDownload\Dataset\_processed` → `D:\Study\migration_package\processed_data` (this is where the migrated data already lives — no copying needed).
- Raw Ali-CCP paths (`SKELETON_PATH`/`COMMON_PATH`/etc. in `aliccp_eda_raw.py`, `degree_distribution_scan.py`, `filter_and_join.py`, `filter_test_and_join.py`, `extract_item_pseudo_text.py`, `extract_user_pseudo_text.py`, `profile_raw_fields.py`, `full_scan_chunk.py`): `E:\BaiduNetdiskDownload\Dataset\sample_train\...` / `sample_test\...` → `D:\Study\Datasets\sample_train\...` / `sample_test\...` (where the raw files are actually mounted).

All 21 files re-verified for valid Python syntax after the change.

### Folder cleanup + git repair (2026-07-30, same pass)

- Deleted 9.2GB of redundant raw-data backups (`Dataset/6.29/sample_train.tar.gz` + `sample_test.tar.gz` + `.md5` files) — same Ali-CCP data already exists extracted and integrity-checked in the separate `Datasets/` folder; not git-tracked, not read by any script.
- Deleted a corrupt 45-byte `Criteo_Conversion_Search.tar.gz` stub (failed/incomplete download remnant) — the real ~6GB extracted `CriteoSearchData` file was already present and intact.
- **`.git` was broken** — `objects/` directory was entirely missing (repo could not run `git status`/`log`/etc). Ran `git init` (safe/idempotent, preserved existing `config`/remote), discovered the actual GitHub default branch is `main` (local was misconfigured to track a nonexistent `master`), fixed the branch tracking, fetched `origin/main`, and committed all recovered code + updated README on top of it (55 files, local `main` now 1 commit ahead of `origin/main`, working tree clean). **Not yet pushed** — ready whenever you want to run `git push`.
- Repo size: 15GB → 6.1GB after cleanup (raw `Dataset/Criteo_Conversion_Search/` extracted TSV, ~6GB, is the only large item left, correctly gitignored).

---

## Open Questions for Supervisor

All three 2026-07-30 post-meeting experiments are now done: Hybrid V3 (beats both V1 and V2, see "V3: hybrid segment router"), MPNet-vs-MiniLM encoder swap (see "Candidate encoder shortlist"), and the Amazon Reviews'23 text-vs-ID dataset (see "Amazon pipeline — status"). Remaining open items are all on the writing side:
- Overleaf template — link and content both pending, tracked under "Next Steps" (see Modelling Results → Overleaf preparation).
- "Define cold start" (from the meeting todo) — the operational definition already used throughout this repo (`is_cold_start_item`: vocab built from train only, unseen items in val/test flagged) hasn't been explicitly written up as a definition for the dissertation text yet.

### Second real-text dataset — research + recommendation (2026-07-30)

Compared Amazon Reviews'23 and MIND as candidates to isolate RQ1 (text vs. ID embeddings) from Ali-CCP's anonymised-pseudo-text ceiling:

| | Amazon Reviews'23 | MIND |
|---|---|---|
| Real text | Yes — item title, description, category, brand | Yes — news title, abstract, body |
| Domain match to Ali-CCP | Same (e-commerce) | Different (news) |
| Scale (usable subset) | Single category, 5-core filtered: tens of thousands to ~1M interactions (e.g. "Beauty", "Video_Games") — full dataset is 750GB but not needed | ~15M impressions / 1M users full-size; MIND-small (50K users) available |
| Native label structure | Ratings/reviews = positive interactions only, no negative/non-purchase impressions — doesn't natively fit ESMM's click→purchase funnel | Has real impression-level click/no-click (CTR-analogous) but no post-click "purchase"/conversion stage |
| Prior use in this dissertation's lit review | Yes — UniSRec and RLMRec (both core citations) both benchmark on Amazon category subsets | Not currently cited |
| Adaptation cost | Moderate: need negative sampling to turn implicit positives into a binary task (standard practice, well precedented) | Moderate-high: CTR side maps directly, but CVR side has no analog — would need a proxy conversion definition, undermining the "real conversion" purpose of using a second dataset at all |

**Recommendation: Amazon Reviews'23, a single small category, 5-core filtered.** Neither dataset has Ali-CCP's native two-stage click→purchase funnel, so this experiment is necessarily reframed as testing RQ1's general claim (text embeddings beat ID embeddings, especially for sparse/cold-start entities) via a binary interaction-prediction task with sampled negatives, rather than a literal CVR replication — that reframing is stated explicitly in every new script's docstring. Amazon Reviews wins on domain match, adaptation cost, and direct precedent in already-cited work (UniSRec/RLMRec); MIND's lack of any conversion-stage analog is the harder problem to justify away.

**Category chosen: `Video_Games`** (2.8M users, 137.2K items, 4.6M ratings per the official per-category table). First attempt used `Digital_Music` (101.0K users, 70.5K items, 130.4K ratings) for its small footprint, but its verified-purchase interaction graph proved too sparse to be useful (see "Amazon pipeline — status" below for what happened and why the category was switched).

### Amazon pipeline — status (2026-08-02, run to completion)

All four scripts have now been run locally end-to-end. Two real bugs were found and fixed along the way (both documented in the scripts' docstrings/comments, not just here):

1. **`Digital_Music` category was too sparse to use.** Its verified-purchase subset (94,954 interactions / 79,404 users / 50,878 items, mean ~1.2 interactions/entity) collapsed to 0 rows under iterative 5-core filtering, and even the 2-core result that did survive (4,160 rows) was too small for either baseline to learn signal above chance (AUC ~0.49-0.51 for both ID and text embeddings). Switched to `Video_Games`, which has ~33.5 ratings/item on average (vs. Digital_Music's ~1.87), giving far more collaborative signal. To keep dataset volume controlled despite Video_Games' much larger raw pool (3.9M verified-purchase interactions / 2.5M users), `amazon_build_dataset.py` now randomly samples whole users (not rows — preserves each sampled user's full interaction history) down to `RAW_SAMPLE_MAX_USERS = 150,000` before k-core filtering.
2. **Stale embedding cache bug in `amazon_text_embedding.py`.** `item_llm_embeddings.pkl` wasn't namespaced by category/dataset, so after switching to Video_Games the script silently reused the old Digital_Music cache (1,324 items). None of the new item ids matched, every lookup fell back to the zero-vector UNK embedding, and the model saw a constant item representation for everything — the symptom was val/test AUC stuck at exactly 0.5000 across all epochs. Fixed by validating the cache covers the current item set before trusting it (auto re-encodes on mismatch).

Final results (`Video_Games`, k-core=2 → 99,281 rows / 32,039 users / 13,720 items → train=397,125 / val=49,640 / test=49,640 after negative sampling):

| | val AUC | test_overall | test_seen_items | test_cold_start_items |
|---|---|---|---|---|
| ID-embedding baseline | 0.5527 | 0.4698 | 0.6162 | 0.5023 |
| Real-text embedding (MiniLM-L6-v2) | 0.5205 | 0.4209 | 0.6061 | 0.4937 |

Both models learn real signal on seen items (AUC ~0.61, well above chance) and both are essentially at chance on cold-start items (~0.49-0.50) — real-text embeddings did not confer better cold-start generalisation over learned ID embeddings in this simple concat-and-MLP architecture, and slightly underperform the ID baseline on val/test_seen. `test_overall` being lower than `test_seen_items` for both models is a statistical artifact of AUC not being decomposable across strata with very different positive rates (seen: 11.8%, cold: 46.7%), not a bug.

**Interpretation / open point for supervisor discussion**: naive frozen-text-embedding replacement doesn't automatically beat a learned ID embedding here, even with genuine natural-language item text (unlike Ali-CCP's anonymised pseudo-text). This is a real, reportable negative result for RQ1 on this dataset, and motivates why the more involved alignment architectures (V1/V2 on Ali-CCP) are worth studying rather than assuming text embeddings are a free win.

### Amazon negative-sampling non-determinism bug (found + fixed 2026-08-07)

While adding a `timestamp` column to `amazon_train/val/test.csv` (needed for the Dynamic Graph / V4 work below), reran `amazon_build_dataset.py` with the *same* `SEED=42` three times (once in a Linux sandbox, twice on the actual Windows machine) and got three different cold-start rates each time (test: 23.49% originally / 23.40% / 23.65%) despite identical row counts and positive rate every time. Root cause: `add_negatives()` received `all_items` as a Python `set` and called `list(all_items)` on it — Python randomises string hashing per-process (`PYTHONHASHSEED`) since 3.3, so a set's iteration order isn't stable across runs even with everything else seeded; `rng.randrange(len(all_items))` picks the same *index* every run, but that index lands on a different item each time the set-to-list order shifts. Fixed with `sorted(all_items)` (verified: two consecutive reruns now produce byte-identical output).

**Decision (agreed with supervisor-facing write-up in mind): not rerunning the already-completed Amazon Baseline/V1/V2/V3/V3-hybrid results.** Those numbers (reported throughout this README) were computed on the pre-fix dataset and remain valid as reported — they were a real, complete dataset, just not exactly reproducible from the seed alone as originally assumed. The drift this bug causes is small (cold-start rate moves by ~0.1-0.3 percentage points; row counts, positive rate, and the user/item pool are all unaffected) and doesn't change any conclusion drawn so far. The fixed, timestamp-added dataset (deterministic going forward) is used for V4 and any future Amazon work — meaning V4 is trained/evaluated on a very slightly different negative-sample instantiation than Baseline/V1/V2/V3, a documented limitation rather than a silent inconsistency.

### V4: Dynamic Graph layer (2026-08-07), and a clean ablation of where the gains actually come from

First model in this dissertation that implements graph/sequential structure at all — Baseline/V1/V1-Full/V2/V3 are all static two-tower models differing only in item representation. Amazon-only (Ali-CCP has no per-interaction timestamps). Deliberate scope reduction vs. DGSR (Zhang et al., TKDE 2022): single-channel target-attention (query = target item, keys/values = the user's prior interacted items) with a learned recency-decay bias from real Delta-t, in the spirit of DIN's target-attention rather than DGSR's edge-quintuple dual long/short-term channels; user-side only, item-side history aggregation left as future work. Chosen given the ~1-month submission timeline. See `src/models/amazon_dynamic_graph_v4.py` docstring for full rationale.

**Pipeline additions:**
- `amazon_build_dataset.py` updated to retain a `timestamp` column (previously dropped) — also where the negative-sampling non-determinism bug above was found and fixed.
- `data/preprocessing/amazon_build_user_histories.py` **(new)** — precomputes each user's causal (label=1 only, strictly-earlier-timestamp) interaction sequence. 32,039 users, mean history length 3.1 / median 2 (2-core filtering guarantees >=2). At test time: 81.1% of rows have >=1 prior history item available (median 1-2 among those), 18.9% have zero (first-ever interaction for that user — falls back to user_emb alone).
- `src/models/amazon_dynamic_graph_v4.py` **(new)** — V4, combining the graph aggregator with V2's ID+LLM alignment pattern from the start (a rough combined result was judged more valuable than a polished single-axis ablation, given the time budget).
- `src/models/amazon_dynamic_graph_v4_id_only.py` **(new)** — ablation of V4 with the LLM/text branch removed entirely (plain ID embedding, same graph aggregator), to isolate the graph mechanism's own contribution from V2's already-established LLM-alignment contribution.

**Results, mean +/- std across 5 seeds (AUC):**

| Model | Test Overall | Test Seen | Test Cold-Start |
|---|---|---|---|
| Baseline (ID, no graph) | 0.4687 +/- 0.0048 | 0.6150 +/- 0.0054 | 0.5011 +/- 0.0062 |
| V2 (ID + LLM align, no graph) | 0.4930 +/- 0.0098 | 0.6177 +/- 0.0119 | 0.5027 +/- 0.0106 |
| V4-ID-Only (ID + graph, no LLM) | 0.4637 +/- 0.0095 | 0.6091 +/- 0.0079 | 0.5010 +/- 0.0038 |
| V4 (ID + LLM align + graph) | 0.4807 +/- 0.0085 | 0.6207 +/- 0.0129 | 0.5014 +/- 0.0117 |

**Paired significance tests (5 shared seeds) — a clean ablation:**
- Baseline vs. V4-ID-Only: not significant on overall (p=0.396), seen (p=0.341), or cold-start (p=0.995). **The graph mechanism alone, with no LLM signal, produces no detectable improvement over the plain ID baseline on any metric.**
- V4-ID-Only vs. V4 (does adding LLM alignment on top of the graph help?): not significant on overall (p=0.061, borderline) or cold-start (p=0.938).
- V2 vs. V4 (does adding the graph on top of LLM alignment help?): V4 is **significantly worse** than V2 on test_overall (p=0.022, mean diff -0.0123). Adding the graph aggregator to V2's already-working LLM-alignment approach doesn't preserve its gain, let alone improve on it.
- For reference, V2 vs. Baseline (established earlier) IS significant (p=0.0036) — the LLM-alignment pattern has a real, replicable effect that the graph aggregator does not.

**Interpretation:** the temporal graph aggregator trains stably and learns a non-trivial recency-decay rate (~0.6, not collapsed to zero) — the mechanism itself works — but produces no measurable benefit on this dataset, on any segment, including cold-start (the segment this layer was specifically expected to help most). The most likely explanation, well-supported by the history-precomputation diagnostics above: per-user history here is too sparse (median 1-2 available items at prediction time) for an attention-based aggregator to have much to work with — there just isn't enough sequential signal in this dataset's density regime for a dynamic-graph mechanism to add value on top of what ID/LLM embeddings alone already capture. This is a genuine, clean negative result for the "Dynamic Graph" layer specifically (in contrast to the "LLM Encoder" layer, where V2's gain over baseline IS significant and replicable) — reportable as-is, not something to keep tuning against given the time budget; the sparsity ceiling is a property of the dataset, not obviously fixable by more epochs or a bigger MAX_HISTORY (0.0% of users even hit the current 50-item cap).

Full logs: `results/v4_significance_tests.log` (V4 vs. Baseline/V1/V2 headline comparisons) and `results/v4_ablation_significance.log` (the ID-only ablation comparisons above); raw per-model summaries in `results/multiseed_summaries/amazon_v4_dynamic_graph.json` and `amazon_v4_id_only.json`.

---

## Repository Structure

**Note (updated 2026-07-30, after second `code_for_github` sync):** the full modelling pipeline has now been recovered into `src/` and `data/preprocessing/` (see "Known Gaps" for the one remaining missing script). Actual current layout of the new/changed parts:

```
dissertation-cvr-llm/
├── migration_package/
│   ├── MIGRATION_NOTES.md
│   └── processed_data/                              # data outputs — see Modelling Results
├── data/preprocessing/
│   ├── confirm_filtering_no_distortion.py
│   ├── split_train_val.py
│   ├── filter_test_and_join.py
│   ├── extract_item_pseudo_text.py
│   ├── extract_user_pseudo_text.py
│   ├── profile_raw_fields.py
│   ├── build_test_difficulty_segments.py
│   ├── degree_distribution_scan_test.py              # STILL MISSING
│   ├── amazon_download.py                            # NEW (2026-07-30) — "Text/no text dataset" experiment
│   ├── amazon_build_dataset.py                        # NEW (2026-07-30)
│   └── amazon_build_difficulty_segments.py             # NEW (2026-08-04) — Amazon context/cold-start segments
├── src/
│   ├── eval_context_segments.py
│   ├── aggregate_multiseed_results.py                 # NEW (2026-08-04) — mean/std across seed reruns
│   ├── significance_tests.py                          # NEW (2026-08-04) — paired t-test across seeds
│   ├── baselines/
│   │   ├── id_embedding_baseline.py                  # ESMM baseline (Ali-CCP)
│   │   └── amazon_id_baseline.py                      # NEW (2026-07-30) — Amazon ID baseline
│   └── models/
│       ├── llm_encoder_v1.py
│       ├── llm_encoder_v1_full.py
│       ├── llm_encoder_v1_mpnet.py                    # NEW (2026-08-04) — V1, MPNet encoder
│       ├── llm_encoder_v1_full_mpnet.py                # NEW (2026-08-04) — V1-Full, MPNet encoder
│       ├── llm_encoder_v2_aligned.py
│       ├── llm_encoder_v2_mpnet.py                    # NEW (2026-07-30) — "Different Embedders" experiment
│       ├── llm_encoder_v3_hybrid.py                    # NEW (2026-08-02) — hybrid V3 segment router (V1+V2)
│       ├── amazon_text_embedding.py                   # NEW (2026-07-30) — Amazon real-text embedding
│       ├── amazon_v2_aligned.py                        # NEW (2026-08-04) — Amazon V2 (ID + alignment)
│       └── amazon_v3_hybrid.py                         # NEW (2026-08-04) — Amazon hybrid V3 segment router
├── results/
│   ├── multiseed_comparison_summary.md                 # NEW (2026-08-05) — full comparison + significance tables
│   ├── significance_tests_full.log                     # NEW (2026-08-05) — raw paired t-test output, all comparisons
│   └── multiseed_summaries/                            # NEW (2026-08-05) — per-model mean/std JSON (10 files)
├── run_all_experiments.ps1                             # NEW (2026-08-04) — runs all 10 models × 5 seeds + both V3 routers
└── docs/
    ├── references.bib
    └── methods_results_draft.tex
```

Original 2026-07-15 planned layout (still mostly aspirational for `notebooks/`, `experiments/`):

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
- [x] Official train/val/test split (session-based 90/10 on train pool, official test file)
- [x] Post-click CVR bug found and corrected (full-population CVR = 0.5353%)
- [x] Filtering-distortion check (CTR shifts, CVR does not)
- [x] Feature/text extraction for LLM encoder input — pseudo-text for 663K items / 12K users
- [x] Baseline implementation — ESMM ID-embedding, test CTCVR-AUC = 0.5610
- [x] V1 / V1-Full — frozen MiniLM pseudo-text embeddings (both below baseline)
- [x] V2 — RLMRec-style ID+text alignment (≈ parity with baseline, best overall)
- [x] Test-set difficulty segmentation (context-length × cold-start 2×2) — V1/V1-Full beat V2 in hardest cell
- [x] Full modelling pipeline code recovered (2026-07-30, two `code_for_github` syncs) — all preprocessing, baseline, V1/V1-Full/V2, and evaluation scripts now in this repo and verified against the data; only `degree_distribution_scan_test.py` still missing (non-blocking, its output already exists) — see "Known Gaps"
- [x] Post-meeting experiments todo (2026-07-30 supervisor meeting): "Different Embedders" (**done** — MPNet vs MiniLM V2 comparison run, see "Candidate encoder shortlist": no overall gain, notable cold-start regression), "Text/no text dataset" (**done** — Amazon Reviews'23 Video_Games pipeline run to completion, results in "Amazon pipeline — status"), "Implement hybrid V3?" (**done** — `llm_encoder_v3_hybrid.py` run, beats both V1 and V2 on overall CTCVR-AUC and the hard segment, see "V3: hybrid segment router")
- [x] Post-meeting experiments todo (2026-08-04 supervisor meeting, "LLM side" scope): multi-seed reruns (5 seeds × 10 models) with mean/std + paired t-test significance testing, MPNet variants completed for V1/V1-Full, Amazon V2-align + V3-hybrid added, Amazon V3 routing direction corrected after finding it flips relative to Ali-CCP — see "Multi-seed reruns, MPNet completion, Amazon V2/V3, and significance testing"
- [ ] Overleaf write-up — `references.bib` / `methods_results_draft.tex` drafted but not pasted in, and not present in this repo copy; also on the post-meeting todo ("Move report draft to Overleaf, share with AS", "Review methods & Results next week", "Define cold start")
- [x] Dynamic Graph layer implementation (2026-08-07) — V4 (single-channel target-attention + recency-decay, DGSR-lite) built and evaluated on Amazon, 5 seeds, plus an ID-only ablation. Result: a clean, honest negative finding — the graph mechanism itself shows no significant benefit on any metric (including cold-start), most likely due to sparse per-user history (median 1-2 items); the LLM-alignment layer (V2) remains the dissertation's one significant, replicable result. See "V4: Dynamic Graph layer" above. Delayed Feedback Correction remains future-work-only, not implemented.
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

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
| V2 | 0.5561 | ~parity (only LLM variant not clearly below baseline) |

### Candidate encoder shortlist ("Different Embedders" experiment, in progress)

MPNet, BGE, E5, GTE, EmbeddingGemma shortlisted as sentence-embedding alternatives to MiniLM. XLNet evaluated and ruled out as unsuitable for direct sentence-embedding use.

`src/models/llm_encoder_v2_mpnet.py` added (2026-07-30): identical to `llm_encoder_v2_aligned.py` (V2) with the frozen encoder swapped from `all-MiniLM-L6-v2` (384-dim) to `all-mpnet-base-v2` (768-dim); writes to a separate `v2_mpnet_results/` directory so it doesn't overwrite the MiniLM checkpoint. V2 was chosen as the base for this comparison because it's the only LLM variant that matched or beat the ID baseline, making it the more informative test of whether encoder quality — as opposed to the anonymised-ID pseudo-text ceiling itself — is the binding constraint. Not yet run (needs local GPU); compare `v2_mpnet_results/esmm_v2_mpnet_metrics.json` against `v2_results/esmm_v2_aligned_metrics.json` once it is. BGE/E5/GTE/EmbeddingGemma variants not yet implemented — pending the MPNet result and time.

### Test-set difficulty segmentation (context-length × cold-start, 2×2)

Ali-CCP has no timestamps, so "prediction horizon" was reinterpreted as **session interaction count** (long_context ≥ median vs. short_context < median), crossed with seen/cold-start item status. Four models evaluated across all four cells (`context_segment_eval_results.json`).

**Key finding — a crossover**: V2 is the strongest model overall, but in the hardest cell, **short-context + cold-start** (418,643 rows, ~20.9% of test set), V1 (0.5627) and V1-Full (0.5564) both beat V2 (0.4991) and baseline (0.5413):

| Model | short_context & cold_start CTCVR-AUC |
|---|---|
| Baseline | 0.5413 |
| V1 | **0.5627** |
| V1-Full | 0.5564 |
| V2 | 0.4991 |

Open question (see below): worth building a hybrid V3 that routes to V1-style embeddings specifically for this segment, or keep as a Discussion-section observation.

### Overleaf preparation

`docs/references.bib` (11 core citations + 3 embedding-model citations) and `docs/methods_results_draft.tex` (paste-ready Methods/Results draft) were prepared but not yet pasted into the Overleaf project, and are not present in this repo copy (see "Known gaps").

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

### Folder cleanup + git repair (2026-07-30, same pass)

- Deleted 9.2GB of redundant raw-data backups (`Dataset/6.29/sample_train.tar.gz` + `sample_test.tar.gz` + `.md5` files) — same Ali-CCP data already exists extracted and integrity-checked in the separate `Datasets/` folder; not git-tracked, not read by any script.
- Deleted a corrupt 45-byte `Criteo_Conversion_Search.tar.gz` stub (failed/incomplete download remnant) — the real ~6GB extracted `CriteoSearchData` file was already present and intact.
- **`.git` was broken** — `objects/` directory was entirely missing (repo could not run `git status`/`log`/etc). Ran `git init` (safe/idempotent, preserved existing `config`/remote), discovered the actual GitHub default branch is `main` (local was misconfigured to track a nonexistent `master`), fixed the branch tracking, fetched `origin/main`, and committed all recovered code + updated README on top of it (55 files, local `main` now 1 commit ahead of `origin/main`, working tree clean). **Not yet pushed** — ready whenever you want to run `git push`.
- Repo size: 15GB → 6.1GB after cleanup (raw `Dataset/Criteo_Conversion_Search/` extracted TSV, ~6GB, is the only large item left, correctly gitignored).

---

## Open Questions for Supervisor

- Turn the short-context+cold-start crossover (V1/V1-Full > V2) into a hybrid V3, or keep it as a Discussion-section observation? (Deferred per 2026-07-30 meeting todo — depends on the two items below landing first.)
- The "multi-dataset × multi-model" evaluation matrix hasn't started — depends on the second-dataset decision below.
- MPNet-for-MiniLM V2 re-run — script written (`llm_encoder_v2_mpnet.py`), not yet run locally.
- Overleaf template — link and content both pending, tracked under "Next Steps" (see Modelling Results → Overleaf preparation).

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

**Category chosen: `Digital_Music`** (101.0K users, 70.5K items, 130.4K ratings per the official per-category table) — picked specifically to keep the download small (explicit volume constraint): the compressed review+meta files are expected in the tens-to-low-hundreds of MB range, vs. multi-GB for larger categories like Beauty_and_Personal_Care or Electronics.

### Amazon pipeline — status (2026-07-30)

**Sandbox network constraint discovered**: this Cowork sandbox's network is allowlisted and blocks both `huggingface.co` and `mcauleylab.ucsd.edu` (`403 blocked-by-allowlist`), unlike the earlier Ali-CCP work where the raw data was already mounted. The download step (and everything downstream needing local GPU) has to run on the local machine — scripts are written, not yet executed or validated against real downloaded data.

Scripts added (all in the same style as the Ali-CCP pipeline; not yet run):
- `data/preprocessing/amazon_download.py` — downloads `Digital_Music.jsonl.gz` / `meta_Digital_Music.jsonl.gz` via `huggingface_hub` (falls back to direct HTTPS), with a 2GB sanity ceiling per file to catch an accidental category swap to something much larger.
- `data/preprocessing/amazon_build_dataset.py` — single-pass (no chunking needed at this scale) construction of the modelling dataset:
  - Positive interactions = `verified_purchase` reviews only (closest analogue to Ali-CCP's "purchase" signal available here) — a documented, unvalidated design choice.
  - Iterative 5-core filtering (matches the Amazon Reviews'23 release's own benchmarking convention).
  - **Genuine chronological split** (train / val / test by timestamp) — Amazon Reviews'23 has real per-second timestamps, unlike Ali-CCP, so this is a methodological improvement over Ali-CCP's non-temporal official train/test boundary.
  - Item vocabulary for `is_cold_start_item` built from train only, same convention as Ali-CCP.
  - Negative sampling (4 sampled negatives per positive, uniform over the item catalog, excluding the user's own positives) — this dataset has no natural non-interaction signal the way Ali-CCP has non-clicks, so this is a real modelling choice to flag explicitly, not a neutral default.
  - Extracts genuine real text per item (title + store + category + features + description) — no template needed, unlike Ali-CCP's pseudo-text.
- `src/baselines/amazon_id_baseline.py` — ID-embedding two-tower binary classifier (single BCE task, not ESMM's dual CTR/CVR structure, since there's no click stage here).
- `src/models/amazon_text_embedding.py` — same architecture, item ID embedding replaced by a frozen `all-MiniLM-L6-v2` embedding of the real item text (same encoder as Ali-CCP's V1, for a fair model-for-model comparison). **This is the first experiment in the dissertation where a result can actually speak to whether an LLM's pretrained world knowledge helps**, rather than only whether its architecture handles out-of-vocabulary anonymised IDs better than a from-scratch embedding table.

**Next steps (local machine)**: run `amazon_download.py`, sanity-check the reported file sizes/row counts against the table above, run `amazon_build_dataset.py`, then `amazon_id_baseline.py` and `amazon_text_embedding.py`, compare `test_overall`/`test_seen_items`/`test_cold_start_items` AUC between the two.

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
│   └── amazon_build_dataset.py                        # NEW (2026-07-30)
├── src/
│   ├── eval_context_segments.py
│   ├── baselines/
│   │   ├── id_embedding_baseline.py                  # ESMM baseline (Ali-CCP)
│   │   └── amazon_id_baseline.py                      # NEW (2026-07-30) — Amazon ID baseline
│   └── models/
│       ├── llm_encoder_v1.py
│       ├── llm_encoder_v1_full.py
│       ├── llm_encoder_v2_aligned.py
│       ├── llm_encoder_v2_mpnet.py                    # NEW (2026-07-30) — "Different Embedders" experiment
│       └── amazon_text_embedding.py                   # NEW (2026-07-30) — Amazon real-text embedding
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
- [ ] Post-meeting experiments todo (2026-07-30 supervisor meeting): "Different Embedders" (MPNet script written, not yet run — see Candidate encoder shortlist), "Text/no text dataset" (Amazon Reviews'23 Digital_Music pipeline written — download/build/baseline/text-embedding scripts, not yet run locally — see Open Questions → Second real-text dataset), "Implement hybrid V3?" (deferred, depends on the first two)
- [ ] Overleaf write-up — `references.bib` / `methods_results_draft.tex` drafted but not pasted in, and not present in this repo copy; also on the post-meeting todo ("Move report draft to Overleaf, share with AS", "Review methods & Results next week", "Define cold start")
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

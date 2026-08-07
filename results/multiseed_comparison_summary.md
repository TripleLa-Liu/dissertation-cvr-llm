# Multi-Seed Comparison Summary (5 seeds: 42, 123, 2026, 7, 99)

_Generated from `aggregate_multiseed_results.py` outputs, 2026-08-05._

## Ali-CCP (CTCVR-AUC, mean ± std across 5 seeds)

| Model | Test Overall | Test Seen Items | Test Cold-Start Items |
|---|---|---|---|
| Baseline (ID) | 0.5562 ± 0.0048 | 0.5572 ± 0.0105 | 0.5573 ± 0.0030 |
| V1 (MiniLM) | 0.5576 ± 0.0248 | 0.5655 ± 0.0372 | 0.5462 ± 0.0127 |
| V1 (MPNet) | 0.5462 ± 0.0133 | 0.5427 ± 0.0188 | 0.5499 ± 0.0126 |
| V1-Full (MiniLM) | 0.5121 ± 0.0011 | 0.4908 ± 0.0019 | 0.5421 ± 0.0031 |
| V1-Full (MPNet) | 0.5111 ± 0.0093 | 0.5051 ± 0.0033 | 0.5187 ± 0.0243 |
| V2 (MiniLM) | 0.5574 ± 0.0050 | 0.5566 ± 0.0052 | 0.5639 ± 0.0089 |
| V2 (MPNet) | 0.5576 ± 0.0134 | 0.5646 ± 0.0109 | 0.5492 ± 0.0247 |

## Amazon Reviews'23 Video_Games (AUC, mean ± std across 5 seeds)

| Model | Test Overall* | Test Seen Items | Test Cold-Start Items |
|---|---|---|---|
| Baseline (ID) | 0.4687 ± 0.0048 | 0.6150 ± 0.0054 | 0.5011 ± 0.0062 |
| V1 (MiniLM, text-replace) | 0.4178 ± 0.0025 | 0.6047 ± 0.0064 | 0.4945 ± 0.0035 |
| V2 (MiniLM, align) | 0.4930 ± 0.0098 | 0.6177 ± 0.0119 | 0.5027 ± 0.0106 |

*See caveat below — Test Overall pools two segments with very different positive rates (seen items 11.8% positive vs. cold-start items 46.7% positive on the same test set), which can and does pull the pooled AUC below both segment-level AUCs. Treat the seen/cold-start columns as the primary comparison for Amazon, not Test Overall.
## Paired Significance Tests (paired t-test across the same 5 seeds)

### Ali-CCP

| Comparison | Metric | Mean diff | t | p | Significant? |
|---|---|---|---|---|---|
| Baseline vs V1 | test_overall.ctcvr_auc | +0.0014 | -0.11 | 0.918 | No |
| Baseline vs V2 | test_overall.ctcvr_auc | +0.0011 | -0.44 | 0.686 | No |
| V1 vs V2 | test_overall.ctcvr_auc | -0.0002 | 0.02 | 0.986 | No |
| Baseline vs V1 | test_cold_start.ctcvr_auc | -0.0111 | 2.12 | 0.101 | No |
| Baseline vs V2 | test_cold_start.ctcvr_auc | +0.0065 | -1.85 | 0.138 | No |
| **V1 vs V2** | **test_cold_start.ctcvr_auc** | **+0.0177** | **-5.65** | **0.0048** | **Yes** |
| Baseline vs V2 | test_seen.ctcvr_auc | -0.0007 | 0.24 | 0.822 | No |
| Baseline vs V1 | test_seen.ctcvr_auc | +0.0083 | -0.41 | 0.702 | No |
| V1-MiniLM vs V1-MPNet | test_overall.ctcvr_auc | -0.0114 | 1.31 | 0.260 | No |
| V2-MiniLM vs V2-MPNet | test_overall.ctcvr_auc | +0.0002 | -0.03 | 0.976 | No |
| V1Full-MiniLM vs V1Full-MPNet | test_overall.ctcvr_auc | -0.0010 | 0.24 | 0.820 | No |

**Takeaway:** across 5 seeds, the only statistically significant difference on Ali-CCP is **V2 beating V1 on the cold-start segment** (p=0.0048) — this is the one result solid enough to state as more than a single-run point estimate. Everything else (Baseline vs V1/V2 overall, MiniLM vs MPNet encoder swap) is not distinguishable from seed noise at n=5. This also means "MPNet doesn't help over MiniLM" and "V1/V1-Full/V2 vs Baseline overall" should be described as directional/non-significant findings in Results, not firm claims.

### Amazon

| Comparison | Metric | Mean diff | t | p | Significant? |
|---|---|---|---|---|---|
| Baseline vs V1 | test_overall.auc | -0.0509 | 18.58 | <0.0001 | Yes* |
| Baseline vs V2 | test_overall.auc | +0.0243 | -6.13 | 0.0036 | Yes* |
| V1 vs V2 | test_overall.auc | +0.0752 | -19.46 | <0.0001 | Yes* |
| Baseline vs V1 | test_cold_start.auc | -0.0066 | 1.66 | 0.172 | No |
| Baseline vs V2 | test_cold_start.auc | +0.0016 | -0.59 | 0.589 | No |
| V1 vs V2 | test_cold_start.auc | +0.0082 | -1.32 | 0.257 | No |
| Baseline vs V2 | test_seen.auc | +0.0027 | -0.45 | 0.676 | No |

*Significant on test_overall, but test_overall is the confounded pooled metric described above — since none of the seen/cold-start subgroup differences are themselves significant, these "significant" overall-AUC gaps most likely reflect how each model's score distribution interacts with the 11.8%-vs-46.7% positive-rate mismatch between segments, not a genuine ranking-quality difference. Recommend not citing the Amazon test_overall significance results in the dissertation without this caveat attached.

## Amazon V3-Hybrid: routing direction fix (2026-08-05)

`amazon_v3_hybrid.py`'s first run mechanically copied Ali-CCP's routing direction
(hard segment -> V1). That direction is wrong for Amazon: per-cell breakdown showed
V2 actually wins short_context & cold_start (0.5173 vs V1's 0.4872), while V1 wins
the other three cells. Flipped the rule (hard segment -> V2, elsewhere -> V1) and
reran. Before/after:

| | overall AUC | short_context & cold_start AUC |
|---|---|---|
| V1 alone | 0.4209 | 0.4872 |
| V2 alone | 0.5088 | 0.5173 |
| V3-hybrid (WRONG direction, first run) | 0.4443 | 0.4872 |
| V3-hybrid (CORRECT direction, rerun) | **0.4944** | **0.5173** |
| Baseline (ID) | 0.4698 | n/a |

Corrected V3-hybrid now: (a) gets the hard-segment cell right (matches V2's 0.5173,
as intended), (b) beats Baseline overall (0.4944 vs 0.4698), and (c) matches or
beats V2 in every individual cell (V1's cells: long&seen 0.6061 vs V2's 0.5970,
long&cold 0.4994 vs V2's 0.4877, short&seen 0.6070 vs V2's 0.5988; hard-segment
cell ties V2 exactly).

**But** the corrected hybrid's *overall/pooled* AUC (0.4944) is still below V2
alone (0.5088), even though it wins-or-ties every individual cell against V2. This
looks contradictory but isn't a bug -- it's the same pooling effect noted above for
`test_overall` (Baseline/V1/V2 comparison): AUC on a pooled set depends on the joint
cross-segment ranking of scores, not just each segment's internal ranking, and the
two segments have very different positive rates (11.7%-12.0% for "seen" cells vs.
46-47% for "cold_start" cells). Swapping V1's cells into V2's changes the
cross-segment score calibration even though within-segment ranking improved
everywhere, which is enough to move the pooled AUC either direction. Net
implication for write-up: on Amazon, prefer citing the per-cell breakdown as the
primary evidence (where the hybrid genuinely wins/ties everywhere), and flag
`overall` AUC on this dataset as a metric to interpret cautiously rather than as
the headline comparison -- unlike Ali-CCP, where the hybrid's `overall` AUC
(0.5649) *did* cleanly beat every individual model.

Only a single seed (checkpoints trained at seed=42); no significance test run on
this cell-level breakdown.

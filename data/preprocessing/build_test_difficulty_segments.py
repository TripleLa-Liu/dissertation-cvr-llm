"""
Ali-CCP test-set easy/hard segmentation by "context length" (2026-07-21)
============================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

ALREADY RUN (2026-07-21) -- this script was executed directly against the
mounted dataset during this session; aliccp_test_with_segments.csv already
exists in WORK_DIR. Kept here for reproducibility / to show the exact
methodology, not because it still needs running.

WHY "SESSION INTERACTION COUNT" INSTEAD OF "PREDICTION HORIZON"
------------------------------------------------------------------
The meeting notes ask to split the test set into easier/harder segments,
"e.g. by prediction horizon length". Ali-CCP carries NO per-row timestamp
(confirmed when choosing the train/test split methodology -- only Criteo,
our secondary dataset, has real delay timestamps), so a literal time-based
prediction-horizon split isn't computable on this dataset.

Reinterpreted proxy: session interaction count -- reusing the EXACT counts
already computed by degree_distribution_scan_test.py (session_counter in
aliccp_degree_counters_test.pkl, the same counts K_SESSION=200 was chosen
from). Rationale: a session with more observed interactions gives the
model more behavioural signal to condition on, a defensible stand-in for
"more context before the prediction point" even without literal
timestamps. This is a DOCUMENTED ASSUMPTION -- flag for supervisor
confirmation, alongside the alternative of a literal time-horizon split
once/if a timestamped second dataset is introduced (see "LLM text
feasibility" note's real-text-dataset discussion).

WHAT THIS SCRIPT DOES
-----------------------
1. Loads session_counter from aliccp_degree_counters_test.pkl (exact
   interaction count per common_feature_index, computed on the FULL
   unfiltered test file).
2. Joins it onto every row of aliccp_test_filtered_joined.csv via
   common_feature_index.
3. Computes the median session count among KEPT rows (all already
   >= K_SESSION=200 by construction of the k-core filter) and labels each
   row long_context (>= median) or short_context (< median).
4. Writes aliccp_test_with_segments.csv = original columns +
   session_interaction_count, context_segment.
5. Prints a sanity-check CTR/CVR breakdown by segment (crossed with the
   existing is_cold_start_item flag).

RESULT (2026-07-21): 2,006,347 rows. session_interaction_count: min=200,
median=243, max=784, mean=265.6. Sanity-check breakdown (CTR / CVR
post-click):
  long_context  (n=1,012,202): CTR=3.4695%  CVR=0.4755%
  short_context (n=994,145):   CTR=3.7080%  CVR=0.4340%
  long_context  & seen       (n=571,017): CVR=0.5561%  (easiest)
  long_context  & cold-start (n=441,185): CVR=0.3927%  (hardest)
  short_context & seen       (n=575,502): CVR=0.4687%
  short_context & cold-start (n=418,643): CVR=0.3945%
Context length shows a real but SECONDARY effect (long_context CVR is
consistently higher than short_context within both seen and cold-start
groups) -- cold-start status remains the dominant difficulty axis. This
gives a genuine 2x2 difficulty matrix (not just the single cold-start
split reported so far) for the per-model evaluation in
eval_context_segments.py.

USAGE (if re-running from scratch)
------------------------------------
python build_test_difficulty_segments.py
"""
import csv
import os
import pickle
import shutil
import statistics

WORK_DIR = r"E:\BaiduNetdiskDownload\Dataset\_processed"
COUNTERS_PATH = os.path.join(WORK_DIR, "aliccp_degree_counters_test.pkl")
TEST_PATH = os.path.join(WORK_DIR, "aliccp_test_filtered_joined.csv")
OUTPUT_PATH = os.path.join(WORK_DIR, "aliccp_test_with_segments.csv")


def main():
    with open(COUNTERS_PATH, "rb") as f:
        counters = pickle.load(f)
    session_counter = counters["session_counter"]
    print(f"Loaded session_counter: {len(session_counter):,} distinct common_feature_index values.")

    with open(TEST_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        rows = list(reader)

    counts = [session_counter.get(r[idx["common_feature_index"]], 0) for r in rows]
    median_count = statistics.median(counts)
    print(f"Rows: {len(rows):,}")
    print(f"session_interaction_count stats: min={min(counts)}, median={median_count}, "
          f"max={max(counts)}, mean={statistics.mean(counts):.1f}")

    stats = {}

    def bump(key, is_click, is_purchase):
        s = stats.setdefault(key, {"n": 0, "click": 0, "purchase_among_click": 0})
        s["n"] += 1
        if is_click:
            s["click"] += 1
            if is_purchase:
                s["purchase_among_click"] += 1

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(header + ["session_interaction_count", "context_segment"])
        for r, cnt in zip(rows, counts):
            segment = "long_context" if cnt >= median_count else "short_context"
            writer.writerow(r + [cnt, segment])

            is_click = r[idx["click"]] == "1"
            is_purchase = r[idx["purchase"]] == "1"
            is_cold = r[idx["is_cold_start_item"]] == "1"
            bump(("overall", None), is_click, is_purchase)
            bump((segment, None), is_click, is_purchase)
            bump((segment, is_cold), is_click, is_purchase)

    print(f"\nWrote {OUTPUT_PATH}")
    print("\nSanity-check breakdown (CTR / CVR post-click):")
    for key in sorted(stats.keys(), key=lambda k: str(k)):
        s = stats[key]
        ctr = s["click"] / s["n"] if s["n"] else 0
        cvr = s["purchase_among_click"] / s["click"] if s["click"] else float("nan")
        print(f"  {key}: n={s['n']:,}  CTR={ctr:.4%}  CVR(post-click)={cvr:.4%}")


if __name__ == "__main__":
    main()

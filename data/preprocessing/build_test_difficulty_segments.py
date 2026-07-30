"""
Ali-CCP test-set easy/hard segmentation by "context length"
============================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

Splits the test set into easier/harder segments by session interaction
count, crossed with the existing cold-start flag, giving a 2x2 difficulty
matrix for per-model evaluation in eval_context_segments.py.

Ali-CCP carries no per-row timestamp (only Criteo, the secondary dataset,
has real delay timestamps), so a literal time-based "prediction horizon"
split isn't computable here. This uses session interaction count as a
proxy instead — reusing the exact counts from degree_distribution_scan_test.py
(session_counter in aliccp_degree_counters_test.pkl, the same counts
K_SESSION=200 was chosen from): a session with more observed interactions
gives the model more behavioural signal to condition on, a defensible
stand-in for "more context" even without literal timestamps. This is a
documented assumption, not a validated one.

Steps:
1. Loads session_counter from aliccp_degree_counters_test.pkl (exact
   interaction count per common_feature_index, on the full unfiltered
   test file).
2. Joins it onto every row of aliccp_test_filtered_joined.csv via
   common_feature_index.
3. Computes the median session count among kept rows (all already
   >= K_SESSION=200 by construction) and labels each row long_context
   (>= median) or short_context (< median).
4. Writes aliccp_test_with_segments.csv = original columns +
   session_interaction_count, context_segment.
5. Prints a CTR/CVR breakdown by segment, crossed with is_cold_start_item.

Result: 2,006,347 rows, session_interaction_count min=200, median=243,
max=784, mean=265.6. Context length shows a real but secondary effect
(long_context CVR consistently higher than short_context within both seen
and cold-start groups) — cold-start status remains the dominant difficulty
axis.
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

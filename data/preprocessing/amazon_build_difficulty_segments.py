"""
Amazon Reviews'23 test-set easy/hard segmentation by "context length"
============================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

Amazon-side counterpart to data/preprocessing/build_test_difficulty_segments.py
(Ali-CCP), needed so amazon_v3_hybrid.py can apply the same
context-length x cold-start routing rule that Ali-CCP's V3 uses (2026-08-04
supervisor request to extend the robustness comparison across both
datasets).

Ali-CCP used session_interaction_count (from the raw degree-scan counters)
as its context-richness proxy, since it has no per-row timestamps. Amazon
Reviews'23 has no equivalent session/common_feature_index concept either.

First attempt used each user's row count in TRAIN only as the proxy, but
that turned out to be degenerate: because the split is chronological,
59.7% of test rows belong to a user_id with ZERO rows in train at all (a
"new" user by the time of the test window), so the median train-only count
over test rows was 0 — a median split at 0 puts every row in
long_context (count >= 0 is always true) and none in short_context. Fixed
by counting each user's total row count across train+val+test COMBINED
instead: since the dataset was built with 2-core filtering (every user has
>= 2 real interactions somewhere in the full dataset), every user has a
nonzero combined count, giving a real distribution (median=15, min=10,
over test rows) to split on. This uses val/test rows to compute a
descriptive stratification variable, not as model input, so it isn't a
train/test leakage concern — the same sense in which is_cold_start_item
itself is also computed by checking test items against the train
vocabulary rather than being something the model is trained on directly.
This is a documented assumption, not a validated one, same caveat as the
Ali-CCP version.

Steps:
1. Loads amazon_train.csv, amazon_val.csv, amazon_test.csv, counts rows
   per user_id across all three combined (total_interaction_count).
2. Joins that count onto every row of amazon_test.csv via user_id.
3. Computes the median total_interaction_count among test rows, labels
   each row long_context (>= median) or short_context (< median).
4. Writes amazon_test_with_segments.csv = original columns +
   total_interaction_count, context_segment.
5. Prints a positive-rate breakdown by segment, crossed with
   is_cold_start_item, mirroring the Ali-CCP script's sanity check.
"""
import os
import statistics

import pandas as pd

WORK_DIR = r"D:\Study\migration_package\processed_data"
PROCESSED_DIR = os.path.join(WORK_DIR, "amazon", "processed")
TRAIN_PATH = os.path.join(PROCESSED_DIR, "amazon_train.csv")
VAL_PATH = os.path.join(PROCESSED_DIR, "amazon_val.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "amazon_test.csv")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "amazon_test_with_segments.csv")


def main():
    print(f"Loading {TRAIN_PATH}, {VAL_PATH}, {TEST_PATH} ...")
    dtype = {"user_id": str, "item_id": str, "label": "int8", "is_cold_start_item": "int8"}
    train_df = pd.read_csv(TRAIN_PATH, dtype=dtype)
    val_df = pd.read_csv(VAL_PATH, dtype=dtype)
    test_df = pd.read_csv(TEST_PATH, dtype=dtype)
    print(f"  train={len(train_df):,} val={len(val_df):,} test={len(test_df):,} rows")

    combined_counts = pd.concat([train_df["user_id"], val_df["user_id"],
                                  test_df["user_id"]]).value_counts()
    print(f"  {len(combined_counts):,} distinct users across all three splits combined")

    test_df["total_interaction_count"] = test_df["user_id"].map(combined_counts).fillna(0).astype(int)
    n_zero = int((test_df["total_interaction_count"] == 0).sum())
    if n_zero:
        print(f"  WARNING: {n_zero:,} test rows have a user_id not found anywhere "
              f"(unexpected — investigate before trusting the split).")

    median_count = statistics.median(test_df["total_interaction_count"].tolist())
    print(f"total_interaction_count stats (over test rows): "
          f"min={test_df['total_interaction_count'].min()}, median={median_count}, "
          f"max={test_df['total_interaction_count'].max()}, "
          f"mean={test_df['total_interaction_count'].mean():.1f}")

    test_df["context_segment"] = test_df["total_interaction_count"].apply(
        lambda c: "long_context" if c >= median_count else "short_context")

    test_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {OUTPUT_PATH}")

    print("\nSanity-check breakdown (positive rate = purchase/interaction rate):")
    is_cold = test_df["is_cold_start_item"].astype(bool)
    segment = test_df["context_segment"]
    label = test_df["label"]

    def bump(name, mask):
        n = int(mask.sum())
        rate = float(label[mask].mean()) if n else float("nan")
        print(f"  {name:28s} n={n:>8,}  positive_rate={rate:.4%}")

    bump("overall", pd.Series(True, index=test_df.index))
    for seg in ("long_context", "short_context"):
        bump(seg, segment == seg)
        bump(f"{seg} & seen", (segment == seg) & (~is_cold))
        bump(f"{seg} & cold_start", (segment == seg) & is_cold)


if __name__ == "__main__":
    main()

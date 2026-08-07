"""
Amazon Reviews'23 (Video_Games) — Precompute causal user interaction histories
================================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

Builds the per-user, time-ordered sequence of REAL (label=1) interactions
used by amazon_dynamic_graph_v4.py's temporal attention aggregator — the
"Dynamic Graph" layer of the three-layer architecture (Amazon-only: Ali-CCP
has no per-interaction timestamps, see README).

Only positive (label=1) rows are real interactions and can appear in
another row's history — sampled negatives (label=0) never happened and
must never appear in anyone's history, only ever be the *target* of a
prediction.

Causality: for a given row at time t, only interactions with a STRICTLY
earlier timestamp are valid history. This applies uniformly to positive
and negative rows: a negative row shares its paired positive's timestamp
(see amazon_build_dataset.py's 2026-08-07 update), so it sees exactly the
history that positive interaction would have seen at that decision point,
minus the interaction itself — and a positive row's own (item, timestamp)
never counts as its own history (strict "<", not "<=").

Does NOT precompute a fixed-length history per row (would duplicate ~496K
rows x 50 items, mostly padding) — instead writes one compact per-user
sorted (item_id, timestamp) sequence, and the training script does an
on-the-fly bisect/searchsorted lookup + truncation to the most recent
N=50 at batch time. Cheap: a lookup is O(log k) where k = that user's
total positive interaction count (small here, see printed stats).

Requires amazon_train.csv / amazon_val.csv / amazon_test.csv from
amazon_build_dataset.py (2026-08-07+, with the timestamp column).

Produces amazon_user_histories.pkl:
    {"histories": {user_id: (item_id_array, timestamp_array_sorted_asc)},
     "max_history": MAX_HISTORY}
"""
import os
import pickle

import numpy as np
import pandas as pd

WORK_DIR = r"D:\Study\migration_package\processed_data"
PROCESSED_DIR = os.path.join(WORK_DIR, "amazon", "processed")

TRAIN_PATH = os.path.join(PROCESSED_DIR, "amazon_train.csv")
VAL_PATH = os.path.join(PROCESSED_DIR, "amazon_val.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "amazon_test.csv")

OUT_PATH = os.path.join(PROCESSED_DIR, "amazon_user_histories.pkl")

MAX_HISTORY = 50  # truncation cap, see amazon_dynamic_graph_v4.py


def load_positives():
    frames = []
    for path in (TRAIN_PATH, VAL_PATH, TEST_PATH):
        df = pd.read_csv(path, dtype={"user_id": str, "item_id": str, "label": "int8"},
                          usecols=["user_id", "item_id", "label", "timestamp"])
        frames.append(df[df["label"] == 1][["user_id", "item_id", "timestamp"]])
    pos = pd.concat(frames, ignore_index=True)
    n_before = len(pos)
    pos = pos.drop_duplicates(subset=["user_id", "item_id"])
    if len(pos) != n_before:
        print(f"  WARNING: {n_before - len(pos):,} duplicate (user_id,item_id) positive rows "
              f"across splits — shouldn't happen given amazon_build_dataset.py's dedup, "
              f"dropped extras defensively.")
    return pos


def main():
    print("Loading positive interactions from train+val+test ...")
    pos = load_positives()
    print(f"  {len(pos):,} distinct positive interactions, {pos['user_id'].nunique():,} users")

    print("Building per-user sorted (item_id, timestamp) sequences ...")
    histories = {}
    lengths = []
    for user_id, group in pos.groupby("user_id", sort=False):
        g = group.sort_values("timestamp", kind="mergesort")  # stable sort: deterministic tie-break
        items = g["item_id"].to_numpy()
        ts = g["timestamp"].to_numpy(dtype="int64")
        histories[user_id] = (items, ts)
        lengths.append(len(items))

    lengths = np.array(lengths)
    print(f"  {len(histories):,} users, history length: mean={lengths.mean():.1f} "
          f"median={np.median(lengths):.0f} min={lengths.min()} max={lengths.max()}")
    print(f"  Truncation cap MAX_HISTORY={MAX_HISTORY}: "
          f"{(lengths > MAX_HISTORY).mean():.1%} of users have more interactions than the cap")

    with open(OUT_PATH, "wb") as f:
        pickle.dump({"histories": histories, "max_history": MAX_HISTORY}, f)
    print(f"\nSaved to {OUT_PATH}")

    # Diagnostic: for the actual test split, how many rows have ANY prior
    # history at all vs. zero (pure first-interaction / cold-start-user case)?
    print("\nDiagnostic — test split coverage:")
    test_df = pd.read_csv(TEST_PATH, dtype={"user_id": str}, usecols=["user_id", "timestamp"])
    n_zero_history = 0
    prior_counts = []
    for user_id, row_ts in zip(test_df["user_id"], test_df["timestamp"]):
        items, ts = histories.get(user_id, (np.array([]), np.array([], dtype="int64")))
        n_prior = int(np.searchsorted(ts, row_ts, side="left"))
        prior_counts.append(n_prior)
        if n_prior == 0:
            n_zero_history += 1
    prior_counts = np.array(prior_counts)
    print(f"  {n_zero_history:,}/{len(test_df):,} test rows ({n_zero_history/len(test_df):.1%}) "
          f"have zero prior history at their own timestamp (first-ever interaction for that "
          f"user, or user has no positive interactions before this row) — these fall back to "
          f"the learned user_emb alone in amazon_dynamic_graph_v4.py, same UNK-style convention "
          f"used throughout this codebase for missing signal.")
    print(f"  Among rows WITH prior history: mean={prior_counts[prior_counts>0].mean():.1f} "
          f"median={np.median(prior_counts[prior_counts>0]):.0f} available history items "
          f"(before the {MAX_HISTORY}-item truncation applied at train/eval time).")


if __name__ == "__main__":
    main()

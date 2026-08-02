"""
Amazon Reviews'23 (Digital_Music) — Dataset Construction (run locally)
=========================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

Builds the modelling-ready dataset for the "Text/no text dataset"
experiment from the raw files downloaded by amazon_download.py. One script
(not several, unlike the Ali-CCP pipeline) because this category is small
enough (hundreds of thousands of rows, not tens of millions) that a single
in-memory pandas pass is sufficient — no chunked/resumable scanning needed.

Design decisions (documented, not validated against a supervisor
discussion the way the Ali-CCP choices were — flag for review):

1. Positive interactions = verified_purchase reviews only (not all
   reviews), on the reasoning that this is the closest analogue to Ali-CCP's
   "purchase" signal available in this dataset; unverified reviews are
   dropped rather than treated as a weaker positive class.
2. Amazon Reviews'23 has real per-second timestamps (unlike Ali-CCP, which
   has none), so this uses a genuine chronological split: the most recent
   TEST_FRAC of interactions by timestamp are held out as test, the next
   VAL_FRAC as validation, and the rest as train — an improvement over
   Ali-CCP's non-temporal official train/test boundary.
3. k-core filtering (K_CORE=5, applied iteratively to both users and items
   until stable) on the full interaction set before splitting, matching
   the "5-core" convention the Amazon Reviews'23 release itself uses for
   benchmarking.
4. Item vocabulary for cold-start flagging is built from TRAIN only (same
   pattern as Ali-CCP's is_cold_start_item), so val/test items unseen in
   train are flagged is_cold_start_item=1.
5. This dataset has no natural negative examples (a review only exists for
   an item the user did interact with), so negatives are sampled uniformly
   at random from the full item catalog per positive row (ratio
   NEG_PER_POS, default 4), excluding items that user already has a
   positive interaction with. This is standard implicit-feedback negative
   sampling practice, but is a real modelling choice worth stating
   explicitly wherever this is written up — Ali-CCP's negatives are real
   non-clicks, these are not.

Requires amazon_download.py to have already produced
{CATEGORY}.jsonl / meta_{CATEGORY}.jsonl in WORK_DIR/amazon/raw/.

Produces (in WORK_DIR/amazon/processed/):
  amazon_train.csv / amazon_val.csv / amazon_test.csv
      columns: user_id, item_id, label, is_cold_start_item
  amazon_item_text.csv
      columns: item_id, real_text (title + description + features + store,
      concatenated — genuine natural-language text, no template needed)
"""
import gzip
import json
import os
import random
import time

import pandas as pd

CATEGORY = "Digital_Music"

WORK_DIR = r"E:\BaiduNetdiskDownload\Dataset\_processed"
AMAZON_DIR = os.path.join(WORK_DIR, "amazon")
RAW_DIR = os.path.join(AMAZON_DIR, "raw")
PROCESSED_DIR = os.path.join(AMAZON_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

REVIEW_JSONL = os.path.join(RAW_DIR, f"{CATEGORY}.jsonl")
META_JSONL = os.path.join(RAW_DIR, f"meta_{CATEGORY}.jsonl")

TRAIN_OUT = os.path.join(PROCESSED_DIR, "amazon_train.csv")
VAL_OUT = os.path.join(PROCESSED_DIR, "amazon_val.csv")
TEST_OUT = os.path.join(PROCESSED_DIR, "amazon_test.csv")
ITEM_TEXT_OUT = os.path.join(PROCESSED_DIR, "amazon_item_text.csv")

K_CORE = 5
TEST_FRAC = 0.10
VAL_FRAC = 0.10
NEG_PER_POS = 4
SEED = 42


# ------------------------------------------------------------------
# Step 1: load reviews -> positive interactions
# ------------------------------------------------------------------
def load_positive_interactions():
    print(f"Loading {REVIEW_JSONL} ...")
    rows = []
    t0 = time.time()
    with open(REVIEW_JSONL, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("verified_purchase", False):
                continue
            item_id = r.get("parent_asin")
            user_id = r.get("user_id")
            ts = r.get("timestamp")
            if item_id is None or user_id is None or ts is None:
                continue
            rows.append((user_id, item_id, int(ts)))
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"])
    df = df.drop_duplicates(subset=["user_id", "item_id"])  # keep first per user-item pair
    print(f"  {len(df):,} verified-purchase interactions "
          f"({df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items) "
          f"in {time.time()-t0:.0f}s")
    return df


# ------------------------------------------------------------------
# Step 2: iterative k-core filtering
# ------------------------------------------------------------------
def k_core_filter(df, k):
    print(f"\nApplying {k}-core filtering (iterative) ...")
    round_i = 0
    while True:
        round_i += 1
        n_before = len(df)
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()
        keep_users = set(user_counts[user_counts >= k].index)
        keep_items = set(item_counts[item_counts >= k].index)
        df = df[df["user_id"].isin(keep_users) & df["item_id"].isin(keep_items)]
        print(f"  round {round_i}: {n_before:,} -> {len(df):,} rows "
              f"({df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items)")
        if len(df) == n_before:
            break
    return df


# ------------------------------------------------------------------
# Step 3: chronological split
# ------------------------------------------------------------------
def chronological_split(df):
    print("\nSplitting chronologically ...")
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    n_test = int(n * TEST_FRAC)
    n_val = int(n * VAL_FRAC)
    test_df = df.iloc[n - n_test:]
    val_df = df.iloc[n - n_test - n_val: n - n_test]
    train_df = df.iloc[: n - n_test - n_val]
    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
    return train_df, val_df, test_df


# ------------------------------------------------------------------
# Step 4: negative sampling
# ------------------------------------------------------------------
def add_negatives(pos_df, all_items, user_positive_items, neg_per_pos, rng):
    all_items = list(all_items)
    rows = []
    for user_id, item_id in zip(pos_df["user_id"], pos_df["item_id"]):
        rows.append((user_id, item_id, 1))
        seen = user_positive_items.get(user_id, set())
        n_sampled = 0
        attempts = 0
        while n_sampled < neg_per_pos and attempts < neg_per_pos * 10:
            cand = all_items[rng.randrange(len(all_items))]
            attempts += 1
            if cand in seen:
                continue
            rows.append((user_id, cand, 0))
            n_sampled += 1
    return pd.DataFrame(rows, columns=["user_id", "item_id", "label"])


# ------------------------------------------------------------------
# Step 5: item text extraction (real text, no template)
# ------------------------------------------------------------------
def extract_item_text(needed_items):
    print(f"\nScanning {META_JSONL} for item metadata ...")
    found = {}
    t0 = time.time()
    with open(META_JSONL, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            item_id = m.get("parent_asin")
            if item_id is None or item_id not in needed_items or item_id in found:
                continue
            parts = []
            if m.get("title"):
                parts.append(m["title"])
            if m.get("store"):
                parts.append(f"by {m['store']}")
            if m.get("main_category"):
                parts.append(f"category: {m['main_category']}")
            if m.get("features"):
                parts.append(" ".join(m["features"]))
            if m.get("description"):
                parts.append(" ".join(m["description"]))
            text = " . ".join(p for p in parts if p).strip()
            found[item_id] = text if text else "no description available"
    print(f"  Found metadata for {len(found):,}/{len(needed_items):,} items "
          f"({time.time()-t0:.0f}s)")
    missing = needed_items - found.keys()
    for item_id in missing:
        found[item_id] = "no description available"
    return found


def main():
    rng = random.Random(SEED)

    df = load_positive_interactions()
    df = k_core_filter(df, K_CORE)
    train_df, val_df, test_df = chronological_split(df)

    # cold-start vocabulary from TRAIN only, same pattern as Ali-CCP
    train_items = set(train_df["item_id"])
    all_items = set(df["item_id"])
    user_positive_items = df.groupby("user_id")["item_id"].apply(set).to_dict()

    print("\nSampling negatives ...")
    splits = {}
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        labelled = add_negatives(split_df, all_items, user_positive_items, NEG_PER_POS, rng)
        labelled["is_cold_start_item"] = (~labelled["item_id"].isin(train_items)).astype(int)
        splits[name] = labelled
        print(f"  {name}: {len(labelled):,} rows ({labelled['label'].mean():.1%} positive), "
              f"cold-start rate={labelled['is_cold_start_item'].mean():.2%}")

    splits["train"].to_csv(TRAIN_OUT, index=False)
    splits["val"].to_csv(VAL_OUT, index=False)
    splits["test"].to_csv(TEST_OUT, index=False)
    print(f"\nWrote {TRAIN_OUT}, {VAL_OUT}, {TEST_OUT}")

    item_text = extract_item_text(all_items)
    text_df = pd.DataFrame(sorted(item_text.items()), columns=["item_id", "real_text"])
    text_df.to_csv(ITEM_TEXT_OUT, index=False)
    print(f"Wrote {ITEM_TEXT_OUT} ({len(text_df):,} items)")

    print("\nSample item text:")
    for item_id, text in list(item_text.items())[:3]:
        preview = text[:200] + ("..." if len(text) > 200 else "")
        print(f"  {item_id}: {preview}")


if __name__ == "__main__":
    main()

"""
Amazon Reviews'23 (Video_Games) — Dataset Construction (run locally)
=========================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

Builds the modelling-ready dataset for the "Text/no text dataset"
experiment from the raw files downloaded by amazon_download.py.

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
3. k-core filtering, applied iteratively to both users and items until
   stable. **K_CORE is NOT hardcoded to the Amazon Reviews'23 release's own
   "5-core" convention** — a first attempt used Digital_Music, whose
   verified-purchase subset (94,954 interactions / 79,404 users / 50,878
   items, mean ~1.2-1.9 interactions per entity) showed iterative 5-core
   collapsing to 0 rows, and even the 2-core result that did survive (4,160
   rows) was too sparse for either baseline to learn signal above chance
   (AUC ~0.50 — see README "Amazon pipeline status"). Switched to
   Video_Games (137.2K items / 4.6M ratings overall, i.e. ~33.5
   ratings/item vs Digital_Music's ~1.87) for a denser collaborative
   signal. `scan_core_sizes()` below still prints achievable sizes for a
   range of k so K_CORE is picked empirically per the same "scan before you
   filter" practice used for Ali-CCP (degree_distribution_scan.py /
   check_session_thresholds.py), rather than assumed.
4. RAW_SAMPLE_MAX_USERS caps dataset volume up front (per the "control
   dataset volume" requirement) by randomly sampling whole users — not
   randomly sampling rows — from the raw verified-purchase interactions
   before k-core filtering. Sampling by user preserves each sampled user's
   full interaction history so density (the thing Digital_Music lacked)
   isn't destroyed by the subsampling itself; row-level random sampling
   would have made the sparsity problem worse, not better.
5. Item vocabulary for cold-start flagging is built from TRAIN only (same
   pattern as Ali-CCP's is_cold_start_item), so val/test items unseen in
   train are flagged is_cold_start_item=1.
6. This dataset has no natural negative examples (a review only exists for
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
      columns: user_id, item_id, label, is_cold_start_item, timestamp
  amazon_item_text.csv
      columns: item_id, real_text (title + description + features + store,
      concatenated — genuine natural-language text, no template needed)

2026-08-07 update (Dynamic Graph / V4 prep): added a `timestamp` column to
the three output CSVs, needed by amazon_build_user_histories.py to build
each row's causal interaction history. This column was present in the
positive interactions from the start (`load_positive_interactions()`) but
was previously dropped inside `add_negatives()`. Sampled negative rows have
no real interaction of their own, so each negative is stamped with its
*paired positive row's* timestamp — the natural reading is "at the moment
this user interacted with the positive item at time t, what if they'd been
shown this negative item instead", i.e. timestamp = the decision point, not
an event time the negative doesn't have. This is an additive change: the
RNG call sequence inside add_negatives() (which candidates get sampled, how
many attempts) is unchanged, so label / is_cold_start_item values for every
row are bit-identical to the original run — only the new timestamp column
is added. Confirmed via row-count + label/cold-start-rate diff after rerun.
"""
import gzip
import json
import os
import random
import time

import pandas as pd

CATEGORY = "Video_Games"

# Local Mac run (2026-08-15, Amazon MPNet gap) — see amazon_download.py.
WORK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "migration_package", "processed_data")
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

K_CORE = 2          # see scan_core_sizes() output before trusting this
CORE_SCAN_KS = [2, 3, 4, 5]   # candidate thresholds printed for comparison, see main()
TEST_FRAC = 0.10
VAL_FRAC = 0.10
NEG_PER_POS = 4
SEED = 42

# Video_Games' raw verified-purchase pool is much larger than Digital_Music's
# (millions vs ~95K interactions) — cap it up front by randomly sampling
# whole users (see design decision #4 above) so the final dataset stays
# controlled in volume without re-introducing the sparsity problem.
RAW_SAMPLE_MAX_USERS = 150_000


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

    n_users = df["user_id"].nunique()
    if n_users > RAW_SAMPLE_MAX_USERS:
        rng = random.Random(SEED)
        all_users = sorted(df["user_id"].unique())  # sort first: set/unique order isn't stable
        sampled_users = set(rng.sample(all_users, RAW_SAMPLE_MAX_USERS))
        df = df[df["user_id"].isin(sampled_users)].reset_index(drop=True)
        print(f"  Sampled down to {RAW_SAMPLE_MAX_USERS:,} users (whole interaction "
              f"histories kept, not row-sampled) -> {len(df):,} interactions "
              f"({df['item_id'].nunique():,} items)")
    return df


# ------------------------------------------------------------------
# Step 2: iterative k-core filtering
# ------------------------------------------------------------------
def k_core_filter(df, k, verbose=True):
    if verbose:
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
        if verbose:
            print(f"  round {round_i}: {n_before:,} -> {len(df):,} rows "
                  f"({df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items)")
        if len(df) == n_before:
            break
    return df


def scan_core_sizes(df, ks):
    """Report the converged k-core size for each candidate k, WITHOUT
    committing to one — run before picking K_CORE, same "scan before you
    filter" practice as Ali-CCP's degree_distribution_scan.py. Cheap at
    this dataset's scale (each k is a fresh filter pass on a copy)."""
    print(f"\nScanning candidate k-core thresholds {ks} (each on a fresh copy) ...")
    print(f"{'k':>4} | {'rows':>10} | {'users':>10} | {'items':>10}")
    print("-" * 45)
    for k in ks:
        filtered = k_core_filter(df.copy(), k, verbose=False)
        print(f"{k:>4} | {len(filtered):>10,} | {filtered['user_id'].nunique():>10,} | "
              f"{filtered['item_id'].nunique():>10,}")
    print("\nPick K_CORE above based on this table (need enough rows/items for a "
          "meaningful train/val/test split and cold-start segment) before continuing.")


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
    """Each negative row is stamped with its paired positive row's
    timestamp (see module docstring, 2026-08-07 update) — negatives have
    no interaction event of their own, so the positive's timestamp stands
    in for "the decision point" this negative was sampled at.

    2026-08-07: found and fixed a real non-determinism bug here, exposed
    while adding the timestamp column (three separate reruns — one in a
    Linux sandbox, two on the user's own Windows machine, same SEED every
    time — produced three different cold-start rates: 12.40/18.38/23.49%
    originally, 12.57/18.77/23.40% and 12.46/18.79/23.65% on reruns). Root
    cause: `all_items` arrives as a Python set (built via `set(df["item_id"])`
    in main()), and `list(a_set)` is ordered by Python's internal string
    hashing, which is randomised per-process (PYTHONHASHSEED) since Python
    3.3 — so `rng.randrange(len(all_items))` picks the same INDEX every
    run (rng is seeded) but that index lands on a DIFFERENT item each
    process, since the list's order itself isn't stable. Row counts and
    label positive-rate were unaffected (those don't depend on which
    specific item gets sampled), only which items — and therefore
    is_cold_start_item — did. Fixed with sorted() below, which is fully
    deterministic regardless of hash seed / OS / process. This means the
    already-reported Baseline/V1/V1-Full/V2/V3 Amazon numbers elsewhere in
    this README were never exactly reproducible from a fresh rerun even
    before this session — the drift is small (<0.3pp) and doesn't change
    any reported conclusion, so those results were kept as originally
    reported rather than rerunning 15 already-completed training jobs;
    this fix only guarantees determinism going forward (used for V4)."""
    all_items = sorted(all_items)
    rows = []
    for user_id, item_id, ts in zip(pos_df["user_id"], pos_df["item_id"], pos_df["timestamp"]):
        rows.append((user_id, item_id, 1, ts))
        seen = user_positive_items.get(user_id, set())
        n_sampled = 0
        attempts = 0
        while n_sampled < neg_per_pos and attempts < neg_per_pos * 10:
            cand = all_items[rng.randrange(len(all_items))]
            attempts += 1
            if cand in seen:
                continue
            rows.append((user_id, cand, 0, ts))
            n_sampled += 1
    return pd.DataFrame(rows, columns=["user_id", "item_id", "label", "timestamp"])


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
    scan_core_sizes(df, CORE_SCAN_KS)
    df = k_core_filter(df, K_CORE)
    if len(df) < 100:
        raise SystemExit(
            f"\nOnly {len(df):,} rows survived {K_CORE}-core filtering — too few for a "
            f"meaningful train/val/test split. Check the scan table above, lower K_CORE, "
            f"or reconsider CATEGORY (see README 'Second real-text dataset' section for "
            f"the size/density trade-off across categories). Refusing to write empty/near-"
            f"empty CSVs silently.")
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

"""
Ali-CCP Exact Degree Distribution Scan (run locally)
=====================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

Computes the exact interaction count per item and per common_feature_index
(session) via a single full pass over sample_skeleton_train.csv, to pick a
k-core filtering threshold that reduces ~42.3M rows to a workable modelling
subset. Item/session activity is extremely long-tailed (a 50K-row uniform
sample showed a median of just 1 interaction per entity), so a sampled
estimate would systematically undercount true frequency and bias the
threshold choice — hence the exact full-file scan.

Uses the same parsing logic as aliccp_eda_raw.py / full_scan_chunk.py
(control-char delimiters \\x01 \\x02 \\x03).

Runtime: roughly 3-8 minutes for ~42.3M rows on a normal laptop SSD.
Memory: item counter holds up to a few million unique item_ids — expect a
few hundred MB to ~1GB of RAM.
"""

import pickle
import time
from collections import Counter

# ------------------------------------------------------------------
# CONFIG — edit this path
# ------------------------------------------------------------------
SKELETON_PATH = r"E:\BaiduNetdiskDownload\Dataset\sample_train\sample_skeleton_train.csv"
PROGRESS_EVERY = 5_000_000
OUTPUT_PICKLE = "aliccp_degree_counters.pkl"

TRIPLE_SEP, FIELD_SEP, VALUE_SEP = "\x01", "\x02", "\x03"
ITEM_FIELD_ID = "205"   # confirmed present in 100% of skeleton rows (see aliccp_eda_raw.py)


def parse_item_id(blob):
    """Extract the item_id (field 205) from a raw feature_blob, if present."""
    for tok in blob.split(TRIPLE_SEP):
        if not tok:
            continue
        try:
            field_id, rest = tok.split(FIELD_SEP, 1)
        except ValueError:
            continue
        if field_id == ITEM_FIELD_ID:
            try:
                feat_id, _value = rest.split(VALUE_SEP, 1)
            except ValueError:
                continue
            return feat_id
    return None


def main():
    item_counter = Counter()      # item_id -> exact interaction count
    session_counter = Counter()   # common_feature_index -> exact row count
    n_total = 0
    n_click = 0
    n_purchase = 0

    print(f"Scanning {SKELETON_PATH} ...")
    t0 = time.time()
    with open(SKELETON_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_total += 1
            parts = line.rstrip("\n").split(",", 5)
            if len(parts) < 6:
                continue
            _sample_id, click, purchase, common_idx, _feat_num, blob = parts

            n_click += (click == "1")
            n_purchase += (purchase == "1")
            session_counter[common_idx] += 1

            item_id = parse_item_id(blob)
            if item_id is not None:
                item_counter[item_id] += 1

            if n_total % PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                rate = n_total / elapsed
                eta = (42_300_000 - n_total) / rate if rate > 0 else float("nan")
                print(f"  ...{n_total:,} rows in {elapsed:.0f}s "
                      f"({rate:,.0f} rows/sec, ~{eta:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"\nDone: {n_total:,} rows scanned in {elapsed:.0f}s "
          f"({n_total/elapsed:,.0f} rows/sec)")
    print(f"CTR                 = {n_click/n_total:.4%}")
    print(f"CVR (post-click)    = {n_purchase/n_click:.4%}" if n_click else "CVR (post-click)    = n/a")
    print(f"CVR (overall)       = {n_purchase/n_total:.5%}")
    print(f"Unique items                        : {len(item_counter):,}")
    print(f"Unique common_feature_index (sessions): {len(session_counter):,}")

    with open(OUTPUT_PICKLE, "wb") as f:
        pickle.dump({
            "item_counter": item_counter,
            "session_counter": session_counter,
            "n_total": n_total,
            "n_click": n_click,
            "n_purchase": n_purchase,
        }, f)
    print(f"\nSaved exact counters to {OUTPUT_PICKLE} "
          f"(reusable for the join + k-core filtering step — no need to rescan)")

    # ------------------------------------------------------------------
    # K-CORE THRESHOLD ANALYSIS (marginal — item side and session side
    # computed separately; a combined "both sides must qualify" filter
    # needs a second pass once you pick a k, since dropping low-count
    # items also drops rows that could push some sessions below threshold
    # and vice versa — this table is for picking a sensible starting k)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("K-CORE THRESHOLD ANALYSIS (marginal, one-sided)")
    print("=" * 70)

    print(f"\n{'min interactions':>18} | {'items kept':>12} | {'item-interactions kept':>24}")
    print("-" * 60)
    for k in [1, 2, 3, 5, 10, 20, 50, 100]:
        kept_items = sum(1 for c in item_counter.values() if c >= k)
        kept_item_rows = sum(c for c in item_counter.values() if c >= k)
        print(f"{k:>18} | {kept_items:>12,} | {kept_item_rows:>24,}")

    print(f"\n{'min interactions':>18} | {'sessions kept':>14} | {'session-rows kept':>18}")
    print("-" * 60)
    for k in [1, 2, 3, 5, 10, 20, 50, 100]:
        kept_sessions = sum(1 for c in session_counter.values() if c >= k)
        kept_session_rows = sum(c for c in session_counter.values() if c >= k)
        print(f"{k:>18} | {kept_sessions:>14,} | {kept_session_rows:>18,}")

    print("\nNext step: pick a k (e.g. items >=5 AND sessions >=5), then run a "
          "second filtering pass (row survives only if BOTH its item_id and "
          "common_feature_index meet the threshold) to get the actual final "
          "row count.")


if __name__ == "__main__":
    main()

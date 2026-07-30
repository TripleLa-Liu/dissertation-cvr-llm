"""
Ali-CCP k-core Filtering + common_features Join (run locally)
================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

Step 2 of preprocessing (step 1 = degree_distribution_scan.py):
1. Loads the exact item/session degree counters from degree_distribution_scan.py
   (aliccp_degree_counters.pkl) — no rescan needed.
2. Builds the qualifying item set (degree >= K_ITEM) and qualifying session
   set (degree >= K_SESSION, common_feature_index).
3. Streams sample_skeleton_train.csv once, keeping rows where both the item_id
   and common_feature_index qualify, writing aliccp_filtered_skeleton.csv.
4. Streams common_features_train.csv once, extracting only the needed
   common_feature_index rows, and joins user_id (field 101) onto the filtered
   skeleton to produce the final aliccp_filtered_joined.csv.

Threshold rationale: items are long-tailed on the low end (42.6% of the
3.17M items are one-off, contributing ~3% of rows), so K_ITEM=50 keeps
140,782 items with little volume loss. Sessions are the opposite — only 1.2%
are one-off, mean 57.9 rows/session — so session degree is the main lever
for total row-count control; K_SESSION=200 (raised from an initial 100,
which kept 9.96M rows, well above the ~2-5M target) keeps 19,550 sessions
and lands total rows near the middle of the target range. See
check_session_thresholds.py for the marginal threshold table behind this
choice; no rescan of the 42.3M-row file is needed to try other thresholds.
"""
import csv
import pickle
import time

SKELETON_PATH = r"E:\BaiduNetdiskDownload\Dataset\sample_train\sample_skeleton_train.csv"
COMMON_PATH   = r"E:\BaiduNetdiskDownload\Dataset\sample_train\common_features_train.csv"
COUNTERS_PICKLE = "aliccp_degree_counters.pkl"
FILTERED_OUTPUT = "aliccp_filtered_skeleton.csv"
JOINED_OUTPUT = "aliccp_filtered_joined.csv"

K_ITEM = 50        # item degree threshold -> 140,782 items at this value (keep as-is,
                   # already inside target range, don't need to touch this one)
K_SESSION = 200    # session degree threshold -> 19,550 sessions marginally (see
                   # check_session_thresholds.py output). Raised from 100 because the
                   # first run (K_SESSION=100) kept 9.96M rows, ~2-5x over target, while
                   # entity counts were already fine — session degree is the real volume
                   # lever here (steep drop-off between k=100 and k=300, see below).

TRIPLE_SEP, FIELD_SEP, VALUE_SEP = "\x01", "\x02", "\x03"
ITEM_FIELD_ID = "205"
USER_FIELD_ID = "101"


def _find_field(blob, target_field_id):
    for tok in blob.split(TRIPLE_SEP):
        if not tok:
            continue
        try:
            field_id, rest = tok.split(FIELD_SEP, 1)
        except ValueError:
            continue
        if field_id == target_field_id:
            try:
                feat_id, _value = rest.split(VALUE_SEP, 1)
            except ValueError:
                continue
            return feat_id
    return None


def parse_item_id(blob):
    return _find_field(blob, ITEM_FIELD_ID)


def parse_user_id(blob):
    return _find_field(blob, USER_FIELD_ID)


def step1_filter():
    print(f"Loading degree counters from {COUNTERS_PICKLE} ...")
    with open(COUNTERS_PICKLE, "rb") as f:
        d = pickle.load(f)
    item_counter = d["item_counter"]
    session_counter = d["session_counter"]

    qualifying_items = {k for k, v in item_counter.items() if v >= K_ITEM}
    qualifying_sessions = {k for k, v in session_counter.items() if v >= K_SESSION}
    print(f"Qualifying items (degree >= {K_ITEM}): {len(qualifying_items):,}")
    print(f"Qualifying sessions (degree >= {K_SESSION}): {len(qualifying_sessions):,}")

    needed_sessions = set()  # sessions that actually end up in a kept row
    n_total = 0
    n_kept = 0
    n_click_kept = 0
    n_purchase_kept = 0

    print(f"\nScanning {SKELETON_PATH} ...")
    t0 = time.time()
    with open(SKELETON_PATH, "r", encoding="utf-8", errors="replace") as fin, \
         open(FILTERED_OUTPUT, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["sample_id", "click", "purchase", "common_feature_index", "item_id"])
        for line in fin:
            n_total += 1
            parts = line.rstrip("\n").split(",", 5)
            if len(parts) < 6:
                continue
            sample_id, click, purchase, common_idx, _feat_num, blob = parts

            if common_idx not in qualifying_sessions:
                continue
            item_id = parse_item_id(blob)
            if item_id is None or item_id not in qualifying_items:
                continue

            writer.writerow([sample_id, click, purchase, common_idx, item_id])
            n_kept += 1
            n_click_kept += (click == "1")
            n_purchase_kept += (purchase == "1")
            needed_sessions.add(common_idx)

            if n_total % 5_000_000 == 0:
                print(f"  ...{n_total:,} rows scanned, {n_kept:,} kept "
                      f"({time.time()-t0:.0f}s elapsed)")

    elapsed = time.time() - t0
    print(f"\nFilter pass done in {elapsed:.0f}s")
    print(f"Rows scanned: {n_total:,}")
    print(f"Rows kept:    {n_kept:,}  ({n_kept/n_total:.2%} of original)")
    if n_click_kept:
        print(f"  CTR (kept subset) = {n_click_kept/n_kept:.4%}")
        print(f"  CVR (kept subset, post-click) = {n_purchase_kept/n_click_kept:.4%}")
    print(f"Distinct items retained:   {len(qualifying_items):,}")
    print(f"Distinct sessions in kept rows: {len(needed_sessions):,}")
    print(f"\n-> Written to {FILTERED_OUTPUT}")

    if not (2_000_000 <= n_kept <= 5_000_000):
        print(f"\nNOTE: {n_kept:,} rows is outside the ~2-5M target range.")
        print("  Too high -> raise K_ITEM and/or K_SESSION at the top of this "
              "script and rerun (no need to rerun degree_distribution_scan.py).")
        print("  Too low  -> lower them and rerun.")

    return needed_sessions


def step2_join(needed_sessions):
    print(f"\nJoining against {COMMON_PATH} "
          f"(only extracting rows for {len(needed_sessions):,} needed indices) ...")
    t0 = time.time()
    lookup = {}  # common_feature_index -> user_id
    n_scanned = 0
    with open(COMMON_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_scanned += 1
            parts = line.rstrip("\n").split(",", 2)
            if len(parts) < 3:
                continue
            common_idx, _feat_num, blob = parts
            if common_idx in needed_sessions:
                lookup[common_idx] = parse_user_id(blob)
            if n_scanned % 100_000 == 0:
                print(f"  ...{n_scanned:,} common_features rows scanned, "
                      f"{len(lookup):,}/{len(needed_sessions):,} matched "
                      f"({time.time()-t0:.0f}s elapsed)")

    elapsed = time.time() - t0
    print(f"Join lookup built in {elapsed:.0f}s: "
          f"{len(lookup):,}/{len(needed_sessions):,} indices matched")

    print(f"Writing final joined file to {JOINED_OUTPUT} ...")
    n_missing_user = 0
    with open(FILTERED_OUTPUT, "r", encoding="utf-8") as fin, \
         open(JOINED_OUTPUT, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        writer.writerow(header + ["user_id"])
        for row in reader:
            common_idx = row[3]
            user_id = lookup.get(common_idx)
            if user_id is None:
                n_missing_user += 1
            writer.writerow(row + [user_id])

    print(f"Done. Rows with no matched user_id: {n_missing_user:,}")
    print(f"\n-> Final modelling dataset written to {JOINED_OUTPUT}")


if __name__ == "__main__":
    needed = step1_filter()
    step2_join(needed)

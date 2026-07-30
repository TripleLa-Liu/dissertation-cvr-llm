"""
Ali-CCP k-core Filtering + common_features Join (run locally)
================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

STEP 2 of preprocessing (step 1 = degree_distribution_scan.py, already run).

WHAT THIS DOES
--------------
1. Loads the exact item/session degree counters produced by
   degree_distribution_scan.py (aliccp_degree_counters.pkl) — no rescan needed.
2. Builds the set of "qualifying" items (degree >= K_ITEM) and "qualifying"
   sessions/common_feature_index (degree >= K_SESSION).
3. Streams sample_skeleton_train.csv ONCE, keeping only rows where BOTH the
   row's item_id and its common_feature_index qualify. Writes the survivors
   to aliccp_filtered_skeleton.csv.
4. Streams common_features_train.csv ONCE, but only extracts rows whose
   common_feature_index is one we actually need (a small subset of the ~730K
   total) — this keeps the join pass cheap despite common_features being an
   8.6GB file with very long rows (mean feature_num ~518).
5. Joins user_id (field 101) onto the filtered skeleton and writes the final
   modelling-ready file: aliccp_filtered_joined.csv.

THRESHOLD CHOICE (see degree_distribution_scan.py's printed K-CORE TABLE for
the full picture; short version):
  - Items are long-tailed on the LOW end: 42.6% of the 3.17M items are
    one-off appearances, contributing only ~3% of total rows. Filtering low-
    degree items shrinks the item vocabulary a lot with little volume loss.
    K_ITEM=50 keeps 140,782 items — inside the dissertation's 50K-200K
    per-entity target.
  - Sessions (common_feature_index, used as a user/context proxy since real
    user_id isn't in the skeleton file — see aliccp_eda_raw.py) are the
    OPPOSITE: only 1.2% are one-off; most already have substantial activity
    (mean 57.9 rows/session). Low thresholds barely reduce row volume — you
    need K_SESSION=100 to meaningfully cut volume (94,969 sessions kept,
    ~15.8M rows on that axis alone), which is also why session threshold is
    the main lever for total row-count control, not just a noise floor.

If the final row count after BOTH filters is still outside the ~2-5M target:
  - Too high  -> raise K_ITEM and/or K_SESSION, rerun.
  - Too low   -> lower them, rerun.
No rescan of the 42.3M-row file is needed to try new thresholds against the
counters — only step 3 (the actual filter pass) needs rerunning.

UPDATE (after first run, K_ITEM=50/K_SESSION=100 -> 9,964,603 rows kept):
row count was ~2-5x over target while item/session entity counts were
already in range, so K_SESSION was raised, not K_ITEM. Extending the session
k-core table past k=100 (see check_session_thresholds.py) revealed a steep
drop-off: sessions kept falls from 94,969 (k=100) to 19,550 (k=200) to 4,904
(k=300) — most sessions cluster in the 100-300 activity range (plausibly
because the 8-day observation window caps how much activity a single
session/context can accumulate). This means the original 50K-200K entity
target and the 2-5M row target can't both be hit exactly — there's no k
where session count stays >=50K AND row count is already <=5M. Prioritising
the row-count target (the more operationally binding one for training
feasibility) and accepting a smaller-but-still-reasonable session/user count
(~10K-20K, which is a normal scale for published GNN recsys work) is the
recommended trade-off. K_SESSION=200 estimated to land the combined
(item+session) row count near the middle of the target range.

USAGE
-----
1. Make sure aliccp_degree_counters.pkl (from degree_distribution_scan.py) is
   in the same folder as this script.
2. Adjust K_ITEM / K_SESSION below if you want to try different thresholds.
3. Run: python filter_and_join.py
   Expected runtime: ~3-4 min for the skeleton filter pass (same order as
   degree_distribution_scan.py) + a few minutes for the common_features join
   (only ~730K rows total, cheap even though individually large).
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

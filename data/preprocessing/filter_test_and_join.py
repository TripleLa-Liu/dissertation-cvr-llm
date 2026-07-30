"""
Ali-CCP TEST Split — Filtering + common_features Join (run locally)
=====================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

STEP 2 of test-set preprocessing (step 1 = degree_distribution_scan_test.py,
already run). Companion to filter_and_join.py, which produced the TRAIN
subset (aliccp_filtered_joined.csv: 3,249,246 rows / 140,782 items /
19,550 sessions).

DESIGN RATIONALE (train/test split — see chat discussion 2026-07-21)
----------------------------------------------------------------------
We use Alibaba's official provided train/test files as our split (rather
than re-cutting the train file ourselves), because:
  1. We couldn't verify from public docs the exact chronological criteria
     Alibaba used (the Tianchi dataset page is JS-rendered, no extractable
     methodology text) — safer to treat the given boundary as the split
     than to guess at a "first N days" cut.
  2. File-size check: sample_skeleton_test.csv (~11.0GB) is essentially the
     same size as sample_skeleton_train.csv (~11.0GB) — this is NOT a small
     last-day holdout, so whatever the partition criterion, it's a
     substantial, independent chunk of the log, which is what we need.
  3. This is also what virtually every published paper using Ali-CCP does,
     since there's no per-row timestamp to construct a custom split from.

Filtering logic — asymmetric by design:
  - ITEM whitelist is inherited from TRAIN (the 140,782 items kept by
    filter_and_join.py's K_ITEM=50 threshold). We do NOT recompute an item
    threshold on the test file itself — a model only has a real embedding
    for items it saw during training, so "does this test-row's item appear
    in the train vocabulary" is the question that actually matters, not
    "how popular was this item within the test file alone."
  - SESSION threshold (K_SESSION) IS recomputed on the test file's own
    common_feature_index degree distribution, because common_feature_index
    is a per-file local index — train's specific session IDs don't exist
    in the test file's ID space, so there's nothing to "inherit" there.
  - Rows whose item is NOT in the train whitelist are KEPT, not dropped,
    and flagged via is_cold_start_item=1. This gives a natural "harder"
    test segment for free (ties directly into the meeting note's "easier
    vs harder test set segments" item) — these are exactly the cases an
    ID-only baseline cannot represent (no trained embedding exists for
    them), whereas an LLM-text-embedding approach potentially can still
    produce a meaningful representation from the item's textual features.
    is_cold_start_item=0 rows are the "seen" / easier segment, directly
    comparable to train.

USAGE
-----
1. Make sure these are both in WORK_DIR (see CONFIG below — the same shared
   folder used by all four preprocessing scripts):
     - aliccp_filtered_joined.csv   (train output, from filter_and_join.py)
     - aliccp_degree_counters_test.pkl (from degree_distribution_scan_test.py)
2. Adjust K_SESSION_TEST below if the printed test-side k-core table from
   degree_distribution_scan_test.py suggests a different value is needed
   (default: reuse 200, same as train, for methodological consistency).
3. Run: python filter_test_and_join.py
"""
import csv
import os
import pickle
import time

SKELETON_TEST_PATH = r"E:\BaiduNetdiskDownload\Dataset\sample_test\sample_skeleton_test.csv"
COMMON_TEST_PATH   = r"E:\BaiduNetdiskDownload\Dataset\sample_test\common_features_test.csv"

# Same WORK_DIR as the other three preprocessing scripts — keep it fixed so
# every input/output file (train's joined csv, both pkls, test's outputs)
# lives in one place and nothing gets lost again.
WORK_DIR = r"E:\BaiduNetdiskDownload\Dataset\_processed"
os.makedirs(WORK_DIR, exist_ok=True)

TRAIN_JOINED_PATH = os.path.join(WORK_DIR, "aliccp_filtered_joined.csv")  # to build the item whitelist
COUNTERS_TEST_PICKLE = os.path.join(WORK_DIR, "aliccp_degree_counters_test.pkl")
FILTERED_TEST_OUTPUT = os.path.join(WORK_DIR, "aliccp_test_filtered_skeleton.csv")
JOINED_TEST_OUTPUT = os.path.join(WORK_DIR, "aliccp_test_filtered_joined.csv")

K_SESSION_TEST = 200   # applied to TEST's own session degree distribution (see rationale above)

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


def load_train_item_whitelist():
    print(f"Loading train item whitelist from {TRAIN_JOINED_PATH} ...")
    items = set()
    with open(TRAIN_JOINED_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        item_col = header.index("item_id")
        for row in reader:
            items.add(row[item_col])
    print(f"Train item whitelist size: {len(items):,} (expect 140,782)")
    return items


def step1_filter(train_items):
    print(f"\nLoading test degree counters from {COUNTERS_TEST_PICKLE} ...")
    with open(COUNTERS_TEST_PICKLE, "rb") as f:
        d = pickle.load(f)
    session_counter_test = d["session_counter"]

    qualifying_sessions = {k for k, v in session_counter_test.items() if v >= K_SESSION_TEST}
    print(f"Qualifying test sessions (degree >= {K_SESSION_TEST}): {len(qualifying_sessions):,}")

    needed_sessions = set()
    n_total = 0
    n_kept = 0
    n_click_kept = 0
    n_purchase_kept = 0
    n_purchase_among_click_kept = 0  # BUGFIX 2026-07-21, see prints below
    n_cold_start = 0
    n_click_cold = 0
    n_purchase_cold = 0
    n_purchase_among_click_cold = 0
    n_click_seen = 0
    n_purchase_seen = 0
    n_purchase_among_click_seen = 0

    print(f"\nScanning {SKELETON_TEST_PATH} ...")
    t0 = time.time()
    with open(SKELETON_TEST_PATH, "r", encoding="utf-8", errors="replace") as fin, \
         open(FILTERED_TEST_OUTPUT, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["sample_id", "click", "purchase", "common_feature_index",
                          "item_id", "is_cold_start_item"])
        for line in fin:
            n_total += 1
            parts = line.rstrip("\n").split(",", 5)
            if len(parts) < 6:
                continue
            sample_id, click, purchase, common_idx, _feat_num, blob = parts

            if common_idx not in qualifying_sessions:
                continue
            item_id = parse_item_id(blob)
            if item_id is None:
                continue

            is_cold = item_id not in train_items
            writer.writerow([sample_id, click, purchase, common_idx, item_id, int(is_cold)])
            n_kept += 1
            is_click = (click == "1")
            is_purchase = (purchase == "1")
            n_click_kept += is_click
            n_purchase_kept += is_purchase
            # BUGFIX 2026-07-21: post-click CVR must use purchases AMONG
            # clicked rows, not all purchase=1 rows — a small number of
            # click=0,purchase=1 anomaly rows in the raw data were silently
            # inflating the old n_purchase_kept/n_click_kept-style ratios.
            is_purchase_among_click = is_click and is_purchase
            n_purchase_among_click_kept += is_purchase_among_click
            if is_cold:
                n_cold_start += 1
                n_click_cold += is_click
                n_purchase_cold += is_purchase
                n_purchase_among_click_cold += is_purchase_among_click
            else:
                n_click_seen += is_click
                n_purchase_seen += is_purchase
                n_purchase_among_click_seen += is_purchase_among_click
            needed_sessions.add(common_idx)

            if n_total % 5_000_000 == 0:
                print(f"  ...{n_total:,} rows scanned, {n_kept:,} kept "
                      f"({time.time()-t0:.0f}s elapsed)")

    elapsed = time.time() - t0
    n_seen = n_kept - n_cold_start
    print(f"\nFilter pass done in {elapsed:.0f}s")
    print(f"Rows scanned: {n_total:,}")
    print(f"Rows kept:    {n_kept:,}  ({n_kept/n_total:.2%} of test file)")
    if n_click_kept:
        print(f"  CTR (kept test subset)            = {n_click_kept/n_kept:.4%}")
        print(f"  CVR (kept test subset, post-click) = {n_purchase_among_click_kept/n_click_kept:.4%}")
    print(f"\n--- Seen vs cold-start item breakdown (the 'easy vs hard' split) ---")
    print(f"Seen items   (in train's 140,782-item vocabulary): {n_seen:,} rows ({n_seen/n_kept:.2%})")
    if n_click_seen:
        print(f"  CTR (seen) = {n_click_seen/n_seen:.4%}" if n_seen else "")
        print(f"  CVR post-click (seen) = {n_purchase_among_click_seen/n_click_seen:.4%}" if n_click_seen else "")
    print(f"Cold-start items (NOT in train vocabulary):        {n_cold_start:,} rows ({n_cold_start/n_kept:.2%})")
    if n_click_cold:
        print(f"  CTR (cold-start) = {n_click_cold/n_cold_start:.4%}" if n_cold_start else "")
        print(f"  CVR post-click (cold-start) = {n_purchase_among_click_cold/n_click_cold:.4%}" if n_click_cold else "")
    print(f"\nDistinct sessions in kept rows: {len(needed_sessions):,}")
    print(f"\n-> Written to {FILTERED_TEST_OUTPUT}")

    print(f"\nFor reference, TRAIN kept subset (corrected 2026-07-21): CTR 3.22%, "
          f"CVR post-click ~0.53% (3,249,246 rows / 140,782 items / 19,550 sessions) "
          f"— compare against the overall test numbers above to check representativeness.")

    return needed_sessions


def step2_join(needed_sessions):
    print(f"\nJoining against {COMMON_TEST_PATH} "
          f"(only extracting rows for {len(needed_sessions):,} needed indices) ...")
    t0 = time.time()
    lookup = {}
    n_scanned = 0
    with open(COMMON_TEST_PATH, "r", encoding="utf-8", errors="replace") as f:
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

    print(f"Writing final joined file to {JOINED_TEST_OUTPUT} ...")
    n_missing_user = 0
    with open(FILTERED_TEST_OUTPUT, "r", encoding="utf-8") as fin, \
         open(JOINED_TEST_OUTPUT, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        # keep is_cold_start_item as last column -> insert user_id before it
        new_header = header[:-1] + ["user_id", "is_cold_start_item"]
        writer.writerow(new_header)
        for row in reader:
            common_idx = row[3]
            is_cold = row[-1]
            user_id = lookup.get(common_idx)
            if user_id is None:
                n_missing_user += 1
            writer.writerow(row[:-1] + [user_id, is_cold])

    print(f"Done. Rows with no matched user_id: {n_missing_user:,}")
    print(f"\n-> Final test dataset written to {JOINED_TEST_OUTPUT}")
    print(f"   Columns: sample_id, click, purchase, common_feature_index, item_id, "
          f"user_id, is_cold_start_item")


if __name__ == "__main__":
    whitelist = load_train_item_whitelist()
    needed = step1_filter(whitelist)
    step2_join(needed)

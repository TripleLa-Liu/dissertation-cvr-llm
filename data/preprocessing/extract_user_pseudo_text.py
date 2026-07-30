"""
Ali-CCP User Pseudo-Text Extraction (run locally)
=====================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

WHY THIS SCRIPT EXISTS
------------------------
Completes the "extract full categorical feature set" todo (item side was
done by extract_item_pseudo_text.py). This does the USER side, using the
8 demographic fields identified in profile_raw_fields.py — these decode
with materially HIGHER confidence than the item-side fields, because their
cardinality AND missingness pattern match Alimama's separately-published
`user_profile.csv` schema field-for-field (see README "LLM text
feasibility" note): 121=cms_segid(97), 122=cms_group_id(13), 124=gender(2),
125=age_level(7), 126=pvalue_level(3, sparse), 127=shopping_level(3),
128=occupation(2), 129=city_tier(4, semi-sparse). Same underlying caveat as
the item side still applies though: all values are anonymised feat_ids
with no public decoder, so this is still template pseudo-text, not real
natural-language profiles.

SECOND BENEFIT: fixes the user-side cold-start gap
-----------------------------------------------------
Baseline #1's train vocabulary only covers 9,074 users (vs. 140,092 items)
— we flagged this as a likely contributor to the val->test generalisation
gap (unseen users in test get the same zero-vector UNK problem unseen
items do). Building text for every user_id across train/val/test (not just
train) fixes this the same way item pseudo-text fixed item cold-start:
every user gets a real embedding derived from their own demographic
fields, not a fallback.

WHAT THIS DOES
---------------
1. Reads user_id from aliccp_train_split.csv/aliccp_val_split.csv/
   aliccp_test_filtered_joined.csv to build the needed set (union).
2. Streams common_features_train.csv ONCE, recording the 8 demographic
   fields for any needed user_id not yet found (first-seen wins — these
   fields should be stable per user across sessions).
3. Streams common_features_test.csv for any still-missing user_ids.
4. Writes user_pseudo_text.csv: user_id, pseudo_text.

Uses the SAME counter-based early-stop pattern as extract_item_pseudo_text.py
(fixed 2026-07-21 after the first version recomputed a full set difference
every row and never finished — see that script's history).

HOW TO RUN
----------
python extract_user_pseudo_text.py
Expected runtime: common_features files are much smaller than skeleton
(~730K / ~similar rows vs 42-43M), so this should be considerably faster
than the item extraction — likely well under a minute per file.
"""
import csv
import os
import time

WORK_DIR = r"E:\BaiduNetdiskDownload\Dataset\_processed"
TRAIN_SPLIT_PATH = os.path.join(WORK_DIR, "aliccp_train_split.csv")
VAL_SPLIT_PATH = os.path.join(WORK_DIR, "aliccp_val_split.csv")
TEST_PATH = os.path.join(WORK_DIR, "aliccp_test_filtered_joined.csv")

COMMON_TRAIN_PATH = r"E:\BaiduNetdiskDownload\Dataset\sample_train\common_features_train.csv"
COMMON_TEST_PATH = r"E:\BaiduNetdiskDownload\Dataset\sample_test\common_features_test.csv"

OUTPUT_PATH = os.path.join(WORK_DIR, "user_pseudo_text.csv")

TRIPLE_SEP, FIELD_SEP, VALUE_SEP = "\x01", "\x02", "\x03"
USER_FIELD_ID = "101"
DEMOGRAPHIC_FIELDS = {
    "121": "micro_segment",     # cms_segid, 97 values — high-cardinality, weaker interpretability
    "122": "segment_group",     # cms_group_id, 13 values — same caveat
    "124": "gender",            # 2 values
    "125": "age_group",         # 7 values
    "126": "spending_power",    # pvalue_level, 3 values, sparse (~41% coverage)
    "127": "shopping_depth",    # shopping_level, 3 values
    "128": "student_status",    # occupation, 2 values
    "129": "city_tier",         # new_user_class_level, 4 values, semi-sparse (~62% coverage)
}


def parse_fields(blob, wanted_field_ids):
    out = {}
    for tok in blob.split(TRIPLE_SEP):
        if not tok:
            continue
        try:
            field_id, rest = tok.split(FIELD_SEP, 1)
            feat_id, _value = rest.split(VALUE_SEP, 1)
        except ValueError:
            continue
        if field_id == USER_FIELD_ID or field_id in wanted_field_ids:
            out[field_id] = feat_id
    return out


def build_needed_users():
    needed = set()
    for path in [TRAIN_SPLIT_PATH, VAL_SPLIT_PATH, TEST_PATH]:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            user_col = header.index("user_id")
            for row in reader:
                val = row[user_col]
                if val and val != "None":
                    needed.add(val)
    print(f"Total distinct users needing pseudo-text (union of train/val/test): {len(needed):,}")
    return needed


def scan_common_features_for_users(path, needed, found, progress_every=200_000):
    # Same counter-based early-stop as extract_item_pseudo_text.py (NOT a
    # per-row set difference — that was the bug that hung for hours there).
    n_total = 0
    n_remaining = len(needed - found.keys())
    t0 = time.time()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_total += 1
            if n_remaining == 0:
                print(f"  All needed users found early — stopping scan of {path} "
                      f"at row {n_total:,}.")
                break
            parts = line.rstrip("\n").split(",", 2)
            if len(parts) < 3:
                continue
            _common_idx, _feat_num, blob = parts
            fields = parse_fields(blob, DEMOGRAPHIC_FIELDS.keys())
            user_id = fields.get(USER_FIELD_ID)
            if user_id is None or user_id not in needed or user_id in found:
                continue
            found[user_id] = fields
            n_remaining -= 1
            if n_total % progress_every == 0:
                elapsed = time.time() - t0
                print(f"  ...{n_total:,} rows scanned, {len(found):,}/{len(needed):,} "
                      f"users found ({elapsed:.0f}s elapsed)")
    print(f"  Scan of {path} done: {n_total:,} rows, {len(found):,}/{len(needed):,} "
          f"users found so far ({time.time()-t0:.0f}s)")


def build_pseudo_text(fields):
    parts = []
    for field_id, label in DEMOGRAPHIC_FIELDS.items():
        val = fields.get(field_id)
        if val is not None:
            token = f"{label}_{val}"
            parts.append(f"{label.replace('_', ' ')} is {token}")
    if not parts:
        return "user with no recorded demographic attributes"
    return "This user's " + ", ".join(parts) + "."


def main():
    needed = build_needed_users()
    found = {}

    print(f"\nScanning {COMMON_TRAIN_PATH} for user attributes ...")
    scan_common_features_for_users(COMMON_TRAIN_PATH, needed, found)

    missing = needed - found.keys()
    if missing:
        print(f"\n{len(missing):,} users not found in train common_features "
              f"(expected — these are test-only users). "
              f"Scanning {COMMON_TEST_PATH} for them ...")
        scan_common_features_for_users(COMMON_TEST_PATH, missing, found)

    still_missing = needed - found.keys()
    print(f"\nFinal coverage: {len(found):,}/{len(needed):,} users "
          f"({len(found)/len(needed):.2%}). Still missing: {len(still_missing):,}")

    print(f"\nWriting {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "pseudo_text"])
        for user_id in needed:
            fields = found.get(user_id, {})
            writer.writerow([user_id, build_pseudo_text(fields)])

    print(f"Done. {OUTPUT_PATH} has {len(needed):,} rows (one per distinct user).")
    print("\nSample pseudo-text (first 5 found users):")
    for user_id, fields in list(found.items())[:5]:
        print(f"  {user_id}: {build_pseudo_text(fields)}")


if __name__ == "__main__":
    main()

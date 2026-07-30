"""
Ali-CCP Item Pseudo-Text Extraction (run locally)
=====================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

Ali-CCP has no real item titles/descriptions — every attribute is an
anonymised numeric feat_id with no published decoder (see
profile_raw_fields.py). This does not produce genuine natural-language item
descriptions the way UniSRec/RLMRec do with real product titles; instead it
builds template sentences from the item-side categorical fields (206 =
category, 207 = shop, 210 = intention_node, 216 = brand), e.g. "item
category is category_8316768, shop is shop_8385719, brand is
brand_9247244." The template labels are real English words a pretrained LM
has seen, but the numeric IDs after them are out-of-vocabulary — a known,
flagged limitation (see README).

Steps:
1. Reads item_id from aliccp_train_split.csv, aliccp_val_split.csv,
   aliccp_test_filtered_joined.csv to build the union set of item_ids
   needing text (includes cold-start test items).
2. Streams sample_skeleton_train.csv once, recording fields 206/207/210/216
   for any needed item_id not yet found.
3. Streams sample_skeleton_test.csv once for any still-missing item_ids
   (pure cold-start items that only appear in the test file).
4. Writes item_pseudo_text.csv: item_id, pseudo_text (one row per item).
"""
import csv
import os
import time

WORK_DIR = r"E:\BaiduNetdiskDownload\Dataset\_processed"
TRAIN_SPLIT_PATH = os.path.join(WORK_DIR, "aliccp_train_split.csv")
VAL_SPLIT_PATH = os.path.join(WORK_DIR, "aliccp_val_split.csv")
TEST_PATH = os.path.join(WORK_DIR, "aliccp_test_filtered_joined.csv")

SKELETON_TRAIN_PATH = r"E:\BaiduNetdiskDownload\Dataset\sample_train\sample_skeleton_train.csv"
SKELETON_TEST_PATH = r"E:\BaiduNetdiskDownload\Dataset\sample_test\sample_skeleton_test.csv"

OUTPUT_PATH = os.path.join(WORK_DIR, "item_pseudo_text.csv")

TRIPLE_SEP, FIELD_SEP, VALUE_SEP = "\x01", "\x02", "\x03"
ITEM_FIELD_ID = "205"
CONTENT_FIELDS = {
    "206": "category",
    "207": "shop",
    "210": "intention node",
    "216": "brand",
}


def parse_fields(blob, wanted_field_ids):
    """Return dict field_id -> feat_id for any of wanted_field_ids present, plus item_id (205)."""
    out = {}
    for tok in blob.split(TRIPLE_SEP):
        if not tok:
            continue
        try:
            field_id, rest = tok.split(FIELD_SEP, 1)
            feat_id, _value = rest.split(VALUE_SEP, 1)
        except ValueError:
            continue
        if field_id == ITEM_FIELD_ID or field_id in wanted_field_ids:
            out[field_id] = feat_id
    return out


def build_needed_items():
    needed = set()
    for path in [TRAIN_SPLIT_PATH, VAL_SPLIT_PATH, TEST_PATH]:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            item_col = header.index("item_id")
            for row in reader:
                needed.add(row[item_col])
    print(f"Total distinct items needing pseudo-text (union of train/val/test): {len(needed):,}")
    return needed


def scan_skeleton_for_items(path, needed, found, progress_every=5_000_000):
    # PERF BUGFIX 2026-07-21: the early-stop check used to recompute
    # `needed - found.keys()` (a full set difference over up to 663K items)
    # on EVERY row — with 42.3M/43.0M rows that's ~10^13 operations, would
    # never finish in practical time. Track a plain integer counter instead
    # so the check is O(1) per row.
    n_total = 0
    n_remaining = len(needed - found.keys())  # computed ONCE, not per row
    t0 = time.time()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_total += 1
            if n_remaining == 0:
                print(f"  All needed items found early — stopping scan of {path} "
                      f"at row {n_total:,}.")
                break
            parts = line.rstrip("\n").split(",", 5)
            if len(parts) < 6:
                continue
            _sid, _click, _purchase, _cidx, _fnum, blob = parts
            fields = parse_fields(blob, CONTENT_FIELDS.keys())
            item_id = fields.get(ITEM_FIELD_ID)
            if item_id is None or item_id not in needed or item_id in found:
                continue
            found[item_id] = fields
            n_remaining -= 1
            if n_total % progress_every == 0:
                elapsed = time.time() - t0
                print(f"  ...{n_total:,} rows scanned, {len(found):,}/{len(needed):,} "
                      f"items found ({elapsed:.0f}s elapsed)")
    print(f"  Scan of {path} done: {n_total:,} rows, {len(found):,}/{len(needed):,} "
          f"items found so far ({time.time()-t0:.0f}s)")


def build_pseudo_text(fields):
    parts = []
    for field_id, label in CONTENT_FIELDS.items():
        val = fields.get(field_id)
        if val is not None:
            token = f"{label.replace(' ', '_')}_{val}"
            parts.append(f"{label} is {token}")
    if not parts:
        return "item with no recorded attributes"
    return "This item's " + ", ".join(parts) + "."


def main():
    needed = build_needed_items()
    found = {}  # item_id -> {field_id: feat_id, ...}

    print(f"\nScanning {SKELETON_TRAIN_PATH} for item attributes ...")
    scan_skeleton_for_items(SKELETON_TRAIN_PATH, needed, found)

    missing = needed - found.keys()
    if missing:
        print(f"\n{len(missing):,} items not found in train skeleton "
              f"(expected — these are cold-start test items). "
              f"Scanning {SKELETON_TEST_PATH} for them ...")
        scan_skeleton_for_items(SKELETON_TEST_PATH, missing, found)

    still_missing = needed - found.keys()
    print(f"\nFinal coverage: {len(found):,}/{len(needed):,} items "
          f"({len(found)/len(needed):.2%}). Still missing: {len(still_missing):,}")

    print(f"\nWriting {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "pseudo_text"])
        for item_id in needed:
            fields = found.get(item_id, {})
            writer.writerow([item_id, build_pseudo_text(fields)])

    print(f"Done. {OUTPUT_PATH} has {len(needed):,} rows (one per distinct item).")
    print("\nSample pseudo-text (first 5 items found):")
    for item_id, fields in list(found.items())[:5]:
        print(f"  {item_id}: {build_pseudo_text(fields)}")


if __name__ == "__main__":
    main()

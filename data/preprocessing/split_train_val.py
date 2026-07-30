"""
Ali-CCP Train / Validation Split (run locally)
================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

STEP 3 of preprocessing (step 1 = degree_distribution_scan.py, step 2 =
filter_and_join.py, both already run -> aliccp_filtered_joined.csv).

WHY THIS SPLIT DESIGN (agreed with supervisor, 2026-07-21)
------------------------------------------------------------
Two-tier design:
  1. Ali-CCP's OFFICIAL sample_skeleton_test.csv is reserved as the final
     held-out test set. It is NOT touched by this script and should not be
     looked at again until final results are reported (standard practice,
     matches the original ESMM paper's benchmarking protocol on this
     dataset).
  2. The already k-core-filtered TRAIN file (aliccp_filtered_joined.csv,
     3,249,246 rows) is split internally into Train / Validation here, for
     all model development, hyperparameter tuning, and early-stopping
     during the baseline + LLM Encoder V1/V2 work. This split is fixed
     (seeded) and reused for every experiment so results stay comparable.

SPLIT UNIT: session (common_feature_index), NOT row.
  Every row belonging to the same common_feature_index shares the same
  common_features blob (context/user-side attributes). Splitting by row
  would put rows from the same session on both sides of the split and leak
  session-level information from train into validation. So we first split
  the ~19,550 unique sessions, then assign every row to whichever side its
  session landed on.

Note on cold items: qualifying items (K_ITEM>=50) are shared across the
whole filtered file, so both Train and Validation should contain mostly
the same item vocabulary (no session-based split affects item coverage).
The genuine cold-start test happens later against the official test file,
which is expected to contain items never seen in this filtered subset --
that is precisely the scenario LLM-derived embeddings are meant to help
with (RQ1), while pure ID-embedding baselines will need a fallback
(e.g. UNK embedding) for unseen items.

USAGE
-----
1. Make sure aliccp_filtered_joined.csv (from filter_and_join.py) is in the
   same folder as this script, or update INPUT_PATH below.
2. Run: python split_train_val.py
   Expected runtime: well under a minute (single pass over 3.25M rows).
"""
import csv
import os
import random

# Same WORK_DIR as degree_distribution_scan.py / filter_and_join.py /
# degree_distribution_scan_test.py / filter_test_and_join.py — keep fixed
# across all preprocessing scripts so nothing gets lost.
WORK_DIR = r"E:\BaiduNetdiskDownload\Dataset\_processed"
os.makedirs(WORK_DIR, exist_ok=True)

INPUT_PATH = os.path.join(WORK_DIR, "aliccp_filtered_joined.csv")
TRAIN_OUTPUT = os.path.join(WORK_DIR, "aliccp_train_split.csv")
VAL_OUTPUT = os.path.join(WORK_DIR, "aliccp_val_split.csv")

VAL_FRACTION = 0.10
RANDOM_SEED = 42


def collect_sessions(path):
    """First pass: find the set of unique common_feature_index values."""
    sessions = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx_col = header.index("common_feature_index")
        for row in reader:
            sessions.add(row[idx_col])
    return sessions, header, idx_col


def split_sessions(sessions, val_fraction, seed):
    sessions = sorted(sessions)  # deterministic order before shuffling
    rng = random.Random(seed)
    rng.shuffle(sessions)
    n_val = int(len(sessions) * val_fraction)
    val_sessions = set(sessions[:n_val])
    train_sessions = set(sessions[n_val:])
    return train_sessions, val_sessions


def write_split(path, header, idx_col, train_sessions, val_sessions):
    stats = {
        "train": {"rows": 0, "click": 0, "purchase": 0, "purchase_among_click": 0,
                  "items": set(), "sessions": set()},
        "val":   {"rows": 0, "click": 0, "purchase": 0, "purchase_among_click": 0,
                  "items": set(), "sessions": set()},
    }
    click_col = header.index("click")
    purchase_col = header.index("purchase")
    item_col = header.index("item_id")

    with open(path, "r", encoding="utf-8", newline="") as fin, \
         open(TRAIN_OUTPUT, "w", encoding="utf-8", newline="") as ftrain, \
         open(VAL_OUTPUT, "w", encoding="utf-8", newline="") as fval:
        reader = csv.reader(fin)
        header_in = next(reader)
        train_writer = csv.writer(ftrain)
        val_writer = csv.writer(fval)
        train_writer.writerow(header_in)
        val_writer.writerow(header_in)

        for row in reader:
            common_idx = row[idx_col]
            if common_idx in train_sessions:
                dest, writer = "train", train_writer
            elif common_idx in val_sessions:
                dest, writer = "val", val_writer
            else:
                continue  # shouldn't happen
            writer.writerow(row)
            s = stats[dest]
            is_click = (row[click_col] == "1")
            is_purchase = (row[purchase_col] == "1")
            s["rows"] += 1
            s["click"] += is_click
            s["purchase"] += is_purchase
            # BUGFIX 2026-07-21: post-click CVR must use purchases AMONG
            # clicked rows, not all purchase=1 rows — see id_embedding
            # baseline results discussion for how this was caught.
            if is_click and is_purchase:
                s["purchase_among_click"] += 1
            s["items"].add(row[item_col])
            s["sessions"].add(common_idx)

    return stats


def print_stats(name, s):
    rows, click, purchase_among_click = s["rows"], s["click"], s["purchase_among_click"]
    print(f"\n{name}:")
    print(f"  Rows: {rows:,}")
    print(f"  Sessions: {len(s['sessions']):,}")
    print(f"  Distinct items: {len(s['items']):,}")
    if rows:
        print(f"  CTR: {click/rows:.4%}")
    if click:
        print(f"  CVR (post-click): {purchase_among_click/click:.4%}")


if __name__ == "__main__":
    print(f"Pass 1: collecting unique sessions from {INPUT_PATH} ...")
    sessions, header, idx_col = collect_sessions(INPUT_PATH)
    print(f"Found {len(sessions):,} unique sessions.")

    train_sessions, val_sessions = split_sessions(sessions, VAL_FRACTION, RANDOM_SEED)
    print(f"Split: {len(train_sessions):,} train sessions / "
          f"{len(val_sessions):,} val sessions "
          f"(target val fraction = {VAL_FRACTION:.0%})")

    print(f"\nPass 2: writing {TRAIN_OUTPUT} and {VAL_OUTPUT} ...")
    stats = write_split(INPUT_PATH, header, idx_col, train_sessions, val_sessions)

    print_stats("TRAIN", stats["train"])
    print_stats("VALIDATION", stats["val"])

    # Sanity check: CTR/CVR should be close between the two splits and close
    # to the full filtered-dataset values (CTR 3.22%, CVR post-click 0.57%)
    # reported in README.md. Large deviations would suggest the split isn't
    # representative and the session shuffle/seed should be revisited.
    print("\nDone. Reminder: sample_skeleton_test.csv (official Ali-CCP test "
          "file) is untouched and reserved as the final held-out test set -- "
          "do not use it during model development.")

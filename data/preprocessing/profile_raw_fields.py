"""
Ali-CCP Raw Field Schema Profiling (run locally)
====================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

Precursor to feature extraction for LLM input text: before extracting the
full categorical feature set (filtered/joined files only carried scalar
item_id/user_id), this profiles what fields actually exist and what they
mean, rather than trusting public write-ups blindly — the delimiter format
itself (\\x01\\x02\\x03) differs from what's commonly documented, and had to
be confirmed by direct byte inspection (see aliccp_eda_raw.py).

Community sources (e.g. torch-rechub's preprocessing script, which does use
the same verified delimiter format) list ~23 field IDs as the sparse/dense
columns, with loose semantic guesses for a few (206=category, 207=shop,
210=intention node, 216=brand, all item-side). Fields 101=user_id and
205=item_id are independently confirmed via byte-level inspection; the rest
are untested hypotheses, and since the dataset is documented elsewhere as
having "109 features" total, that list of 23 is likely a known subset, not
the full vocabulary. This script empirically profiles all field IDs
actually present in the raw files, so feature choices for pseudo-text are
based on evidence (cardinality, value patterns) rather than borrowed
assumptions.

Takes a reservoir sample from sample_skeleton_train.csv (row-level:
click/purchase/item-side/context fields) and common_features_train.csv
(session-level: user-side fields, mean feature_num ~518). For each field_id
encountered, reports occurrence rate, distinct value count (capped), and
example values — enough to identify which fields are categorical IDs worth
turning into text vs. noise/dense crossed features.
"""
import random
import time
from collections import defaultdict

SKELETON_PATH = r"D:\Study\Datasets\sample_train\sample_skeleton_train.csv"
COMMON_PATH = r"D:\Study\Datasets\sample_train\common_features_train.csv"

SAMPLE_SIZE = 100_000
MAX_DISTINCT_TRACKED = 200   # cap per-field distinct-value tracking (memory)
MAX_EXAMPLES = 8
PROGRESS_EVERY = 2_000_000

TRIPLE_SEP, FIELD_SEP, VALUE_SEP = "\x01", "\x02", "\x03"

# The community-cited "known" field list, for cross-reference only — NOT
# assumed correct, just annotated in the output so it's easy to compare.
KNOWN_FIELDS = {
    "101": "hypothesis: user_id [CONFIRMED via byte inspection]",
    "205": "hypothesis: item_id [CONFIRMED via byte inspection]",
    "206": "hypothesis: item category_id (unverified)",
    "207": "hypothesis: item shop_id (unverified)",
    "210": "hypothesis: item intention_node_id (unverified)",
    "216": "hypothesis: item brand_id (unverified)",
    "301": "hypothesis: position feature (unverified)",
    "508": "hypothesis: combo(109_14, 206) (unverified)",
    "509": "hypothesis: combo(110_14, 207) (unverified)",
    "702": "hypothesis: user-item brand combo (unverified)",
    "121": "hypothesis: user demographic field (unverified)",
    "122": "hypothesis: user demographic field (unverified)",
    "124": "hypothesis: user demographic field (unverified)",
    "125": "hypothesis: user demographic field (unverified)",
    "126": "hypothesis: user demographic field (unverified)",
    "127": "hypothesis: user demographic field (unverified)",
    "128": "hypothesis: user demographic field (unverified)",
    "129": "hypothesis: user demographic field (unverified)",
    "853": "hypothesis: unknown combo feature (unverified)",
    "109_14": "hypothesis: dense user behaviour count (unverified)",
    "110_14": "hypothesis: dense user behaviour count (unverified)",
    "127_14": "hypothesis: dense user behaviour count (unverified)",
    "150_14": "hypothesis: dense user behaviour count (unverified)",
}


def parse_blob(blob):
    """Return dict field_id -> value for every field in the blob (not just one)."""
    out = {}
    for tok in blob.split(TRIPLE_SEP):
        if not tok:
            continue
        try:
            field_id, rest = tok.split(FIELD_SEP, 1)
            feat_id, value = rest.split(VALUE_SEP, 1)
        except ValueError:
            continue
        # feat_id is itself often the "category value id"; value is often a
        # weight/score (frequently "1" for one-hot indicator fields, but not
        # always) — we record BOTH so we can see the pattern per field.
        out.setdefault(field_id, []).append((feat_id, value))
    return out


def reservoir_sample_lines(path, n_sample):
    """Reservoir-sample n_sample raw lines from a large file, single pass."""
    reservoir = []
    n_seen = 0
    t0 = time.time()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_seen += 1
            if len(reservoir) < n_sample:
                reservoir.append(line)
            else:
                j = random.randint(0, n_seen - 1)
                if j < n_sample:
                    reservoir[j] = line
            if n_seen % PROGRESS_EVERY == 0:
                print(f"  ...{n_seen:,} lines seen ({time.time()-t0:.0f}s elapsed)")
    print(f"  Done: {n_seen:,} lines seen, sampled {len(reservoir):,} ({time.time()-t0:.0f}s)")
    return reservoir


def profile_lines(lines, blob_col_index, label):
    # split with maxsplit=blob_col_index so the blob (last field) is kept
    # whole even if it happens to contain a literal comma — matches the
    # split(",", 5) / split(",", 2) pattern used in the other preprocessing
    # scripts (aliccp_eda_raw.py, filter_and_join.py, etc.)
    field_occurrence = defaultdict(int)
    field_values = defaultdict(set)
    n_rows = 0
    for line in lines:
        parts = line.rstrip("\n").split(",", blob_col_index)
        if len(parts) <= blob_col_index:
            continue
        blob = parts[blob_col_index]
        n_rows += 1
        fields = parse_blob(blob)
        for field_id, pairs in fields.items():
            field_occurrence[field_id] += 1
            for feat_id, _value in pairs:
                if len(field_values[field_id]) < MAX_DISTINCT_TRACKED:
                    field_values[field_id].add(feat_id)

    print(f"\n{'='*90}\n{label}: profiled {n_rows:,} sampled rows\n{'='*90}")
    print(f"{'field_id':>10} | {'present in':>10} | {'distinct (capped)':>18} | note")
    print("-" * 90)
    for field_id in sorted(field_occurrence, key=lambda k: -field_occurrence[k]):
        occ = field_occurrence[field_id]
        n_distinct = len(field_values[field_id])
        capped = f"{n_distinct}{'+' if n_distinct >= MAX_DISTINCT_TRACKED else ''}"
        note = KNOWN_FIELDS.get(field_id, "")
        examples = list(field_values[field_id])[:MAX_EXAMPLES]
        print(f"{field_id:>10} | {occ:>10,} ({occ/n_rows:.1%}) | {capped:>18} | {note}")
        print(f"{'':>10}   examples: {examples}")
    return field_occurrence, field_values


def main():
    print(f"Sampling {SAMPLE_SIZE:,} rows from {SKELETON_PATH} ...")
    skeleton_lines = reservoir_sample_lines(SKELETON_PATH, SAMPLE_SIZE)
    # skeleton columns: sample_id,click,purchase,common_feature_index,feature_num,blob -> blob is index 5
    profile_lines(skeleton_lines, blob_col_index=5, label="SKELETON (row-level: item/context fields)")

    print(f"\nSampling {SAMPLE_SIZE:,} rows from {COMMON_PATH} ...")
    common_lines = reservoir_sample_lines(COMMON_PATH, SAMPLE_SIZE)
    # common_features columns: common_feature_index,feature_num,blob -> blob is index 2
    profile_lines(common_lines, blob_col_index=2, label="COMMON_FEATURES (session-level: user/context fields)")

    print("\nDone.")


if __name__ == "__main__":
    main()

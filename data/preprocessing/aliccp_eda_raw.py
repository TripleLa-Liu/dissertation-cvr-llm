"""
Ali-CCP Raw-Format EDA Script
=============================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

The official Ali-CCP files are not fixed-width CSVs: each line has a variable
number of comma-separated fields because the trailing feature list's length
depends on feature_num, so pandas.read_csv() with a fixed column count
misparses this file. This script parses it manually.

Confirmed file format (verified via byte-level inspection of the raw files):
  sample_skeleton_{train,test}.csv:
      sample_id,click,purchase,common_feature_index,feature_num,<feature_blob>
  common_features_{train,test}.csv:
      common_feature_index,feature_num,<feature_blob>

  <feature_blob> uses ASCII control characters as delimiters (not comma/colon
  as commonly documented elsewhere):
      \x01 (SOH)  separates each field:feat_id:value triple from the next
      \x02 (STX)  separates field_id from feat_id within a triple
      \x03 (ETX)  separates feat_id from feature_value within a triple
  e.g. b'101\x0231319\x031.0\x01125\x023438774\x031.0\x01...' decodes to
  field 101 -> feat_id 31319, value 1.0; field 125 -> feat_id 3438774, ...

  common_feature_index lets many skeleton rows share one row of user/context
  features in the common_features file instead of repeating them, which is
  why common_features_train.csv has far fewer rows than the skeleton file.

Field IDs (confirmed on a 500k skeleton / 20k common_features sample):
  - field 205 (item_id): present in 100% of skeleton rows.
  - field 101 (user_id): present in 0% of skeleton rows, but present in
    common_features rows — user-side features live only in
    common_features_{train,test}.csv, keyed by common_feature_index, and are
    shared across every skeleton row with that index. Attaching user_id to a
    skeleton row requires joining on common_feature_index (not done here by
    default, since it needs a full pass over the multi-GB common_features
    file); this script instead reports common_feature_index reuse as a proxy
    for distinct user/context sessions, and item_id sparsity directly.
"""

import re
import time
from collections import Counter
from itertools import islice

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["font.size"] = 11

# ------------------------------------------------------------------
# CONFIG — edit these two paths to point at your extracted files
# ------------------------------------------------------------------
SKELETON_PATH = r"E:\BaiduNetdiskDownload\Dataset\sample_train\sample_skeleton_train.csv"
COMMON_PATH   = r"E:\BaiduNetdiskDownload\Dataset\sample_train\common_features_train.csv"

# Reference (measured on this dissertation's data, via a 2GB byte-sampled
# linecount extrapolation): sample_skeleton_train.csv ~41M rows,
# common_features_train.csv ~730K rows. feature_num averages ~14 per skeleton
# row and ~518 per common_features row (much longer — user profiles are verbose).
# common_features rows are ~5x slower per row to parse than skeleton rows.

SAMPLE_ROWS = 500_000     # rows to sample from skeleton — ~5s at ~100k rows/sec
COMMON_SAMPLE_ROWS = 50_000  # rows to sample from common_features — ~20s at ~2.5k rows/sec
                              # (each row has far more feature triples to parse)

USER_FIELD_ID = "101"   # confirmed present in common_features, NOT in skeleton
ITEM_FIELD_ID = "205"   # confirmed present in 100% of skeleton rows sampled


# ============================================================
# 1. RAW LINE SNIFF — look at real rows before trusting anything
# ============================================================

def sniff_file(path, n=3, label=""):
    print(f"\n--- Raw sniff: {label or path} (first {n} lines) ---")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(islice(f, n)):
            preview = line.strip()
            if len(preview) > 400:
                preview = preview[:400] + " ...[truncated]"
            print(f"[{i}] {preview}")


# ============================================================
# 2. LINE PARSER
# ============================================================

TRIPLE_SEP = "\x01"   # separates feature triples from each other
FIELD_SEP  = "\x02"   # separates field_id from feat_id within a triple
VALUE_SEP  = "\x03"   # separates feat_id from feature_value within a triple


def _parse_feature_blob(blob):
    """Split a raw feature_blob into a list of (field_id, feat_id, value) triples."""
    triples = []
    for tok in blob.split(TRIPLE_SEP):
        if not tok:
            continue
        try:
            field_id, rest = tok.split(FIELD_SEP, 1)
            feat_id, value = rest.split(VALUE_SEP, 1)
        except ValueError:
            continue
        triples.append((field_id, feat_id, value))
    return triples


def parse_skeleton_line(line):
    """Parse one line of sample_skeleton_*.csv into a flat dict."""
    sample_id, click, purchase, common_idx, feat_num, blob = \
        line.rstrip("\n").split(",", 5)
    feat_num = int(feat_num)

    user_id, item_id = None, None
    for field_id, feat_id, _value in _parse_feature_blob(blob):
        if field_id == USER_FIELD_ID:
            user_id = feat_id
        elif field_id == ITEM_FIELD_ID:
            item_id = feat_id

    return {
        "sample_id": sample_id,
        "click": int(click),
        "purchase": int(purchase),
        "common_feature_index": common_idx,
        "feature_num": feat_num,
        "user_id": user_id,
        "item_id": item_id,
    }


def parse_common_line(line, field_counter=None):
    """Parse one line of common_features_*.csv. Optionally tally field_id
    frequency into `field_counter` (a collections.Counter) for diagnostics."""
    common_idx, feat_num, blob = line.rstrip("\n").split(",", 2)
    feat_num = int(feat_num)

    fields_seen = []
    for field_id, _feat_id, _value in _parse_feature_blob(blob):
        fields_seen.append(field_id)
        if field_counter is not None:
            field_counter[field_id] += 1

    return {
        "common_feature_index": common_idx,
        "feature_num": feat_num,
        "field_ids": fields_seen,
    }


# ============================================================
# 3. SAMPLE LOADERS
# ============================================================

def load_skeleton_sample(path=SKELETON_PATH, nrows=SAMPLE_ROWS):
    print(f"\nLoading Ali-CCP skeleton sample ({nrows:,} rows) from:\n  {path}")
    t0 = time.time()
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in islice(f, nrows):
            rows.append(parse_skeleton_line(line))
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


def load_common_sample(path=COMMON_PATH, nrows=COMMON_SAMPLE_ROWS):
    print(f"\nLoading Ali-CCP common_features sample ({nrows:,} rows) from:\n  {path}")
    t0 = time.time()
    rows = []
    field_counter = Counter()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in islice(f, nrows):
            rows.append(parse_common_line(line, field_counter))
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    return df, field_counter


# ============================================================
# 4. EDA
# ============================================================

def eda_aliccp_raw(skeleton_df, common_df=None, field_counter=None):
    print("\n" + "=" * 60)
    print("ALI-CCP (raw format) — Exploratory Data Analysis")
    print("=" * 60)

    n = len(skeleton_df)
    ctr = skeleton_df["click"].mean()
    clicked = skeleton_df[skeleton_df["click"] == 1]
    cvr_post_click = clicked["purchase"].mean() if len(clicked) else float("nan")
    cvr_overall = skeleton_df["purchase"].mean()

    print(f"\n[1] Sampled rows: {n:,}")
    print(f"[2] CTR (click / impression): {ctr:.4%}")
    print(f"[3] CVR (purchase / click, post-click): {cvr_post_click:.4%}")
    print(f"[4] CVR (purchase / impression, overall): {cvr_overall:.4%}")
    print(f"    Clicks: {skeleton_df['click'].sum():,}  "
          f"Purchases: {skeleton_df['purchase'].sum():,}")

    print(f"\n[5] feature_num distribution:\n{skeleton_df['feature_num'].describe()}")

    # user/item extraction coverage — tells you if USER_FIELD_ID/ITEM_FIELD_ID
    # assumption actually matched anything in this sample
    user_hit_rate = skeleton_df["user_id"].notna().mean()
    item_hit_rate = skeleton_df["item_id"].notna().mean()
    print(f"\n[6] Field-ID extraction coverage (sanity check):")
    print(f"    user_id (field {USER_FIELD_ID}) found in {user_hit_rate:.1%} of skeleton rows "
          f"(expected 0% — user features live in common_features, not skeleton; see module docstring)")
    print(f"    item_id (field {ITEM_FIELD_ID}) found in {item_hit_rate:.1%} of skeleton rows")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 1. impression/click/conversion funnel
    n_imp = (skeleton_df["click"] == 0).sum()
    n_click_only = ((skeleton_df["click"] == 1) & (skeleton_df["purchase"] == 0)).sum()
    n_conv = (skeleton_df["purchase"] == 1).sum()
    axes[0].bar(["Impression\n(no click)", "Click\n(no purchase)", "Purchase"],
                [n_imp, n_click_only, n_conv],
                color=["#BDBDBD", "#90CAF9", "#A5D6A7"])
    axes[0].set_yscale("log")
    axes[0].set_title("Sample Distribution (log scale)")
    axes[0].set_ylabel("Count")

    # 2. item sparsity (field 205, confirmed present in 100% of skeleton rows)
    if item_hit_rate > 0:
        item_activity = skeleton_df.dropna(subset=["item_id"]).groupby("item_id").size()
        axes[1].hist(item_activity.clip(upper=item_activity.quantile(0.99)),
                     bins=40, color="#90CAF9", edgecolor="white")
        axes[1].set_title("Interactions per Item (field 205, extracted)")
        axes[1].set_xlabel("Number of interactions")
        print(f"\n[7] Item sparsity (extracted, sample-level):")
        print(f"    Unique items in sample: {item_activity.shape[0]:,}")
        print(f"    Median interactions/item: {item_activity.median():.0f}")
    else:
        axes[1].hist(skeleton_df["feature_num"], bins=30, color="#90CAF9", edgecolor="white")
        axes[1].set_title("feature_num per skeleton row")
        axes[1].set_xlabel("feature_num")

    # 3. common_feature_index reuse — proxy for user/context grouping,
    #    works regardless of field-id assumption correctness
    reuse = skeleton_df.groupby("common_feature_index").size()
    axes[2].hist(reuse.clip(upper=reuse.quantile(0.99)),
                 bins=40, color="#FFCC80", edgecolor="white")
    axes[2].set_title("Rows sharing same common_feature_index")
    axes[2].set_xlabel("Reuse count")
    print(f"\n[8] common_feature_index reuse (proxy for user/context grouping):")
    print(f"    Unique common_feature_index in sample: {reuse.shape[0]:,}")
    print(f"    Median reuse: {reuse.median():.0f}, Mean reuse: {reuse.mean():.1f}")

    plt.suptitle("Ali-CCP Dataset — Key Statistics (raw-format parse)",
                 fontsize=13, fontweight="bold")
    plt.tight_l
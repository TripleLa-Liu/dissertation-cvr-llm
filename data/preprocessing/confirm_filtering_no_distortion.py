"""
Confirm (or refute) whether k-core filtering distorted the label distribution
================================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

WHY THIS SCRIPT EXISTS
------------------------
Supervisor (Dr. Sinclair) asked, re: the k-core filtering summary ("CTR/CVR
of the filtered subset close to the exact full-population values — filtering
didn't distort the label distribution"): "Is there some way to confirm this
with a plot/stats? or is it random sampling?"

This runs two analyses to answer that properly instead of asserting it:
1. Two-proportion z-tests (full population vs. k-core filtered subset) for
   CTR and for CVR (post-click), with 95% Wilson confidence intervals.
2. A degree-bucketed breakdown of CTR/CVR using the existing UNBIASED 50K
   reservoir sample (Algorithm R, from full_scan_chunk.py) joined against the
   exact session_counter (from degree_distribution_scan.py) -- this shows
   *why* any shift happens (k-core selects on session degree specifically),
   rather than just reporting a before/after number.

RESULT (already run once, 2026-07-21/22, against this exact dataset):
  CTR:  full=3.8871%  filtered=3.2192%  -> z=60.4, p<1e-300 (SIGNIFICANT,
        ~17% relative drop -- filtering is NOT random sampling w.r.t. CTR;
        the degree-bucketed breakdown shows CTR decreases as session degree
        increases, roughly monotonically from ~10% at degree 2-9 down to
        ~3.6-3.9% at degree >=100, before k-core's own K_SESSION=200 cutoff
        even applies -- so selecting higher-degree sessions mechanically
        selects lower-CTR sessions too)
  CVR (post-click): full=0.5353%  filtered=0.5277%  -> z=0.33, p=0.74 (NOT
        significant -- the actual dissertation target metric IS preserved
        by filtering; the original claim holds for CVR specifically, just
        not for CTR)

CONCLUSION: the original blanket claim ("filtering didn't distort the label
distribution") was too strong as stated -- it holds for CVR (the metric
that actually matters for this dissertation) but not for CTR, where the
degree-based (non-random) selection mechanism causes a real, explainable,
statistically significant shift.

HOW TO RUN
-----------
python confirm_filtering_no_distortion.py
Requires: aliccp_degree_counters.pkl (from degree_distribution_scan.py) and
aliccp_filtered_joined.csv (from filter_and_join.py) in WORK_DIR, plus the
exact full-population counts already established via full_scan_chunk.py
(hardcoded below as FULL_N/FULL_CLICK/FULL_PURCHASE_AMONG_CLICK -- these are
exact, not estimates, from the resumable full scan of all 42,300,135 rows).
Also needs scan_reservoir.pkl (the 50K unbiased reservoir sample saved by
full_scan_chunk.py) for the degree-bucketed breakdown plot.
Produces: results/figures/filtering_distortion_check.png
"""
import csv
import math
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORK_DIR = r"E:\BaiduNetdiskDownload\Dataset\_processed"
COUNTERS_PATH = os.path.join(WORK_DIR, "aliccp_degree_counters.pkl")
FILTERED_PATH = os.path.join(WORK_DIR, "aliccp_filtered_joined.csv")
RESERVOIR_PATH = os.path.join(WORK_DIR, "scan_reservoir.pkl")   # see note below if missing
FIG_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                       "results", "figures", "filtering_distortion_check.png")

# Exact full-population counts from full_scan_chunk.py's completed scan of all
# 42,300,135 rows of sample_skeleton_train.csv (not estimates).
FULL_N = 42_300_135
FULL_CLICK = 1_644_256
FULL_PURCHASE_AMONG_CLICK = 8_802   # corrected count, see "Statistics bug" note in README


def wilson_ci(x, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = x / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return p, max(0.0, center - half), center + half


def two_prop_ztest(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return p1, p2, z, p_value


def count_filtered_subset():
    n_total = n_click = n_pac = 0
    with open(FILTERED_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        ci, pi = header.index("click"), header.index("purchase")
        for row in reader:
            n_total += 1
            is_click = row[ci] == "1"
            is_purchase = row[pi] == "1"
            if is_click:
                n_click += 1
                if is_purchase:
                    n_pac += 1
    return n_total, n_click, n_pac


def main():
    print("Counting exact filtered-subset click/purchase totals ...")
    n2, click2, pac2 = count_filtered_subset()
    print(f"  filtered: n={n2:,} click={click2:,} purchase_among_click={pac2:,}")

    print("\n=== CTR: full population vs. filtered subset ===")
    p1, p2, z, pval = two_prop_ztest(FULL_CLICK, FULL_N, click2, n2)
    print(f"CTR_full={p1:.4%}  CTR_filtered={p2:.4%}  "
          f"abs diff={abs(p1-p2):.4%}pts  rel diff={abs(p1-p2)/p1:.2%}")
    print(f"two-proportion z-test: z={z:.2f}, p={pval:.3e}")

    print("\n=== CVR (post-click): full population vs. filtered subset ===")
    p1c, p2c, zc, pvalc = two_prop_ztest(FULL_PURCHASE_AMONG_CLICK, FULL_CLICK, pac2, click2)
    print(f"CVR_full={p1c:.4%}  CVR_filtered={p2c:.4%}  "
          f"abs diff={abs(p1c-p2c):.4%}pts  rel diff={abs(p1c-p2c)/p1c:.2%}")
    print(f"two-proportion z-test: z={zc:.4f}, p={pvalc:.4f}")

    # ---- degree-bucketed breakdown (requires the unbiased reservoir sample) ----
    if not os.path.exists(RESERVOIR_PATH):
        print(f"\nNOTE: {RESERVOIR_PATH} not found -- skipping the degree-bucketed "
              "plot (only the aggregate z-tests above were computed). Copy "
              "scan_reservoir.pkl into WORK_DIR to enable it.")
        return

    with open(COUNTERS_PATH, "rb") as f:
        session_counter = pickle.load(f)["session_counter"]
    with open(RESERVOIR_PATH, "rb") as f:
        reservoir = pickle.load(f)

    bins = [(1, 1), (2, 9), (10, 49), (50, 99), (100, 199), (200, 299), (300, 499), (500, 10**9)]
    labels = ['1', '2-9', '10-49', '50-99', '100-199', '200-299', '300-499', '500+']

    def bucket_idx(d):
        for k, (lo, hi) in enumerate(bins):
            if lo <= d <= hi:
                return k
        return None

    n_arr, click_arr, pac_arr = [0] * len(bins), [0] * len(bins), [0] * len(bins)
    for row in reservoir:
        k = bucket_idx(session_counter.get(row["common_feature_index"], 0))
        if k is None:
            continue
        n_arr[k] += 1
        if row["click"]:
            click_arr[k] += 1
            if row["purchase"]:
                pac_arr[k] += 1

    ctr_p, ctr_lo, ctr_hi = [], [], []
    for n, c in zip(n_arr, click_arr):
        p, lo, hi = wilson_ci(c, n)
        ctr_p.append(p * 100); ctr_lo.append((p - lo) * 100); ctr_hi.append((hi - p) * 100)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    ax = axes[0]
    x = np.arange(len(labels))
    ax.bar(x, ctr_p, yerr=[ctr_lo, ctr_hi], capsize=4, color="#4C72B0", alpha=0.85)
    for i, n in enumerate(n_arr):
        ax.annotate(f"n={n}", (x[i], ctr_p[i] + ctr_hi[i] + 0.4), ha='center', fontsize=8, color='dimgray')
    ax.axvline(x=4.5, color='red', linestyle='--', linewidth=1.5, label='K_SESSION=200 cutoff')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Session interaction count (bucket)")
    ax.set_ylabel("CTR (%)")
    ax.set_ylim(0, 26)
    ax.set_title("CTR vs. session degree\n(50K unbiased reservoir sample)")
    ax.legend(fontsize=9, loc='upper right')

    ax = axes[1]
    ctr1, ctr1_lo, ctr1_hi = wilson_ci(FULL_CLICK, FULL_N)
    ctr2, ctr2_lo, ctr2_hi = wilson_ci(click2, n2)
    cvr1, cvr1_lo, cvr1_hi = wilson_ci(FULL_PURCHASE_AMONG_CLICK, FULL_CLICK)
    cvr2, cvr2_lo, cvr2_hi = wilson_ci(pac2, click2)

    labels2 = ["CTR\nfull pop.", "CTR\nfiltered", "CVR\nfull pop.", "CVR\nfiltered"]
    vals = [ctr1 * 100, ctr2 * 100, cvr1 * 100, cvr2 * 100]
    errs_lo = [(ctr1 - ctr1_lo) * 100, (ctr2 - ctr2_lo) * 100, (cvr1 - cvr1_lo) * 100, (cvr2 - cvr2_lo) * 100]
    errs_hi = [(ctr1_hi - ctr1) * 100, (ctr2_hi - ctr2) * 100, (cvr1_hi - cvr1) * 100, (cvr2_hi - cvr2) * 100]
    colors = ["#4C72B0", "#DD8452", "#4C72B0", "#DD8452"]
    xpos = [0, 0.8, 2.2, 3.0]
    ax.bar(xpos, vals, yerr=[errs_lo, errs_hi], capsize=5, color=colors, width=0.6)
    ax.set_xticks(xpos); ax.set_xticklabels(labels2, fontsize=9)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 5.2)
    ax.set_title("Full population vs. k-core filtered subset\n(95% Wilson CI)", pad=14)
    ax.text(0.4, 4.55, f"z={z:.1f}, p<1e-300\n(significant, ~17%\nrelative drop)",
            ha='center', fontsize=8, color='darkred')
    ax.text(2.6, 1.35, f"z={zc:.2f}, p={pvalc:.2f}\n(not significant)",
            ha='center', fontsize=8, color='darkgreen')

    plt.tight_layout()
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    plt.savefig(FIG_OUT, dpi=150)
    print(f"\nSaved plot to {FIG_OUT}")


if __name__ == "__main__":
    main()

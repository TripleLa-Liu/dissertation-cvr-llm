"""
Paired significance testing between two models across matched seeds
============================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

2026-08-04 supervisor request: confirm whether a performance difference
between two model variants (e.g. V2 vs Baseline, V3-Hybrid vs V2) is
statistically significant, not just a single-run difference that could be
noise from random initialisation / data shuffling.

This runs a paired t-test across seeds: for each seed present in BOTH
models' multi-seed metrics files, take that seed's test AUC value from
each model, then test whether the paired differences have a non-zero
mean. Paired (not an independent two-sample test) because the same seed
drives the same train/val/test shuffling and initialisation scheme for
both models — pairing by seed controls for that shared source of
variance, which is the more appropriate test here than an unpaired
two-sample t-test would be.

Requires scipy (pip install scipy; also added to requirements.txt).
Needs >= 2 shared seeds to run at all, but a paired t-test with only 2-3
observations has very low statistical power — 5+ shared seeds recommended
before treating the p-value as a firm claim (see the CAUTION printed below
for n < 5).

Note on what this test does NOT do: it does not account for the fact that
AUC estimates from the SAME test set are themselves correlated across
models (that would call for a DeLong test on a single run's predictions).
This script instead treats each seed's final test AUC as one independent
observation and compares the two distributions of per-seed AUCs — exactly
the "std + t-test across reruns" the supervisor asked for, not a
within-run ranking-correlation test.

Usage:
    python significance_tests.py \
        --a "D:/Study/migration_package/processed_data/baseline_results/esmm_id_baseline_metrics*.json" --a-label Baseline \
        --b "D:/Study/migration_package/processed_data/v2_results/esmm_v2_aligned_metrics*.json" --b-label V2 \
        --group test_overall --metric ctcvr_auc
"""
import argparse
import glob
import json

try:
    from scipy import stats
except ImportError:
    raise SystemExit("Please install scipy: pip install scipy")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="glob pattern for model A's metrics JSON files")
    p.add_argument("--a-label", required=True)
    p.add_argument("--b", required=True, help="glob pattern for model B's metrics JSON files")
    p.add_argument("--b-label", required=True)
    p.add_argument("--group", default="test_overall",
                   help="one of: val, test_overall, test_seen_items, test_cold_start_items")
    p.add_argument("--metric", default="ctcvr_auc",
                   help="one of: ctr_auc, cvr_auc_post_click, ctcvr_auc, auc")
    return p.parse_args()


def load_by_seed(pattern):
    by_seed = {}
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            data = json.load(f)
        seed = data.get("seed", "42_legacy")
        by_seed[seed] = data
    return by_seed


def main():
    args = parse_args()
    a_runs = load_by_seed(args.a)
    b_runs = load_by_seed(args.b)

    shared_seeds = sorted(set(a_runs) & set(b_runs), key=str)
    if len(shared_seeds) < 2:
        raise SystemExit(
            f"Only {len(shared_seeds)} shared seed(s) between {args.a_label} "
            f"(seeds found: {sorted(a_runs, key=str)}) and {args.b_label} "
            f"(seeds found: {sorted(b_runs, key=str)}). Rerun BOTH models under "
            f"the same set of seeds (e.g. --seed 42, 123, 2026, 7, 99 for both) "
            f"before a paired test is possible."
        )

    a_vals, b_vals = [], []
    for seed in shared_seeds:
        a_v = a_runs[seed].get(args.group, {}).get(args.metric)
        b_v = b_runs[seed].get(args.group, {}).get(args.metric)
        if a_v is None or b_v is None:
            print(f"  (skipping seed={seed}: {args.group}.{args.metric} missing in one model's results)")
            continue
        a_vals.append(a_v)
        b_vals.append(b_v)

    n = len(a_vals)
    if n < 2:
        raise SystemExit(f"Only {n} usable paired observation(s) after filtering — need >= 2.")

    print(f"Comparing {args.a_label} vs {args.b_label} on {args.group}.{args.metric}, "
          f"paired across {n} shared seed(s): {shared_seeds}")
    print(f"  {args.a_label}: {a_vals}")
    print(f"  {args.b_label}: {b_vals}")

    t_stat, p_value = stats.ttest_rel(a_vals, b_vals)
    diffs = [b - a for a, b in zip(a_vals, b_vals)]
    mean_diff = sum(diffs) / len(diffs)

    print(f"\nMean difference ({args.b_label} - {args.a_label}): {mean_diff:+.5f}")
    print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
    if n < 5:
        print(f"CAUTION: only n={n} paired seeds — a paired t-test with this few "
              f"observations has very low statistical power; treat this p-value as "
              f"indicative only, not a firm significance claim. 5+ seeds recommended.")
    if p_value < 0.05:
        print(f"-> Difference is statistically significant at alpha=0.05.")
    else:
        print(f"-> Difference is NOT statistically significant at alpha=0.05 "
              f"(consistent with the gap being noise from seed variance alone).")


if __name__ == "__main__":
    main()

"""
Aggregate multi-seed results — mean/std across seed reruns
============================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

2026-08-04 supervisor request: report mean +/- std across multiple random
seeds instead of a single-run point estimate, for a more robust comparison
table. Every training script (id_embedding_baseline.py, llm_encoder_v1.py,
llm_encoder_v1_full.py, llm_encoder_v2_aligned.py, llm_encoder_v2_mpnet.py,
amazon_id_baseline.py, amazon_text_embedding.py) now accepts --seed and
writes seed-suffixed metrics files (e.g. esmm_v2_aligned_metrics_seed123.json)
for any seed other than the original default (42, unsuffixed). This script
reads all matching files for one model, and reports mean/std per metric.

Usage (after rerunning one script under several seeds, e.g.
--seed 42 (default), --seed 123, --seed 2026, --seed 7, --seed 99):

    python aggregate_multiseed_results.py \
        --pattern "D:/Study/migration_package/processed_data/v2_results/esmm_v2_aligned_metrics*.json" \
        --label "V2 (MiniLM)"

Prints mean +/- std for every AUC-type metric found under val/test_overall/
test_seen_items/test_cold_start_items, and writes a combined
{label}_multiseed_summary.json next to the matched files. Works for both
the Ali-CCP ESMM scripts (ctr_auc/cvr_auc_post_click/ctcvr_auc) and the
Amazon single-task scripts (auc) since it just reports whichever keys
are actually present.
"""
import argparse
import glob
import json
import os
import statistics

METRIC_GROUPS = ["val", "test_overall", "test_seen_items", "test_cold_start_items"]
METRIC_KEYS = ["ctr_auc", "cvr_auc_post_click", "ctcvr_auc", "auc"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", required=True,
                   help="glob pattern matching one model's metrics JSON files across seeds, "
                        "e.g. '.../esmm_v2_aligned_metrics*.json'")
    p.add_argument("--label", required=True,
                   help="human-readable name for this model, for the printed table")
    p.add_argument("--out", default=None,
                   help="output summary JSON path (default: alongside matched files)")
    return p.parse_args()


def load_all(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No files matched pattern: {pattern}")
    runs = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        # scripts written/rerun before the 2026-08-04 --seed patch never
        # wrote a "seed" key; those are all the original seed=42 single run.
        seed = data.get("seed", "42 (legacy, pre --seed patch)")
        runs.append((path, seed, data))
    return runs


def collect_values(runs, group, key):
    vals = []
    for _path, _seed, data in runs:
        v = data.get(group, {}).get(key)
        if v is not None:
            vals.append(v)
    return vals


def mean_std(vals):
    if len(vals) == 0:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def main():
    args = parse_args()
    runs = load_all(args.pattern)
    print(f"Found {len(runs)} run(s) for '{args.label}':")
    for path, seed, _ in runs:
        print(f"  seed={seed}: {os.path.basename(path)}")
    if len(runs) < 2:
        print("\nWARNING: only one run found — std will be 0.0 and not meaningful. "
              "Rerun the training script under additional --seed values first "
              "(e.g. --seed 123, --seed 2026, --seed 7, --seed 99) before trusting "
              "the std reported here.")

    summary = {"label": args.label, "n_seeds": len(runs), "seeds": [s for _, s, _ in runs]}

    print(f"\n{'metric':40s} {'mean':>10s} {'std':>10s}   (n={len(runs)})")
    print("-" * 65)
    for group in METRIC_GROUPS:
        summary[group] = {}
        for key in METRIC_KEYS:
            vals = collect_values(runs, group, key)
            if not vals:
                continue
            mean, std = mean_std(vals)
            summary[group][key] = {"mean": mean, "std": std, "n": len(vals), "values": vals}
            print(f"{group + '.' + key:40s} {mean:10.4f} {std:10.4f}")

    out_path = args.out
    if out_path is None:
        first_dir = os.path.dirname(runs[0][0])
        safe_label = args.label.replace(" ", "_").replace("(", "").replace(")", "")
        out_path = os.path.join(first_dir, f"{safe_label}_multiseed_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()

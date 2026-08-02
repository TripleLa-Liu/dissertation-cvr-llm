"""
Amazon Reviews'23 — Category Download (run locally)
=====================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

Downloads one category's review + item-metadata files from the Amazon
Reviews'23 release (McAuley Lab, UCSD), for the "Text/no text dataset"
experiment — a second, real-text dataset used to isolate RQ1 (does
replacing ID embeddings with text embeddings help) from Ali-CCP's
anonymised-pseudo-text ceiling.

Category chosen: Video_Games (2.8M users, 137.2K items, 4.6M ratings per
the official per-category statistics table). First attempt used
Digital_Music (101.0K users, 70.5K items, 130.4K ratings) for its small
footprint, but its verified-purchase interaction graph turned out too
sparse to be useful: mean ~1.2 interactions/user collapsed to 0 rows under
even a 4-core filter, and the 2-core dataset it did produce (4,160 rows)
was too small for either baseline to learn signal above chance (AUC ~0.50
for both ID and text embeddings — see README "Amazon pipeline status").
Video_Games has ~33.5 ratings/item on average (vs ~1.87 for Digital_Music)
giving much more collaborative signal to learn from, while still being two
orders of magnitude smaller than the largest categories (Books, Clothing,
Home_and_Kitchen), keeping the "control dataset volume" constraint
reasonable. Change CATEGORY below to try a different one — see the
per-category table at https://amazon-reviews-2023.github.io/main.html
before doing so, to check its size first.

Tries huggingface_hub first (the officially recommended access method),
falling back to a direct HTTPS download if huggingface_hub isn't
installed. Reports the downloaded file sizes and row counts at the end —
check these against the expected scale above before proceeding to
amazon_build_dataset.py, in case a much larger category was selected by
mistake.
"""
import gzip
import json
import os
import shutil
import time
import urllib.request

CATEGORY = "Video_Games"

WORK_DIR = r"D:\Study\migration_package\processed_data"
AMAZON_DIR = os.path.join(WORK_DIR, "amazon")
RAW_DIR = os.path.join(AMAZON_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

REVIEW_URL = f"https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/{CATEGORY}.jsonl.gz"
META_URL = f"https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_{CATEGORY}.jsonl.gz"
HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"
HF_REVIEW_PATH = f"raw/review_categories/{CATEGORY}.jsonl.gz"
HF_META_PATH = f"raw/meta_categories/meta_{CATEGORY}.jsonl.gz"

REVIEW_GZ = os.path.join(RAW_DIR, f"{CATEGORY}.jsonl.gz")
META_GZ = os.path.join(RAW_DIR, f"meta_{CATEGORY}.jsonl.gz")
REVIEW_JSONL = os.path.join(RAW_DIR, f"{CATEGORY}.jsonl")
META_JSONL = os.path.join(RAW_DIR, f"meta_{CATEGORY}.jsonl")

# Sanity-check ceiling: abort if either downloaded file is bigger than this
# (catches an accidental category swap to something much larger, e.g.
# Books or Electronics, before it eats disk/bandwidth). Video_Games'
# review file is ~35x Digital_Music's by rating count, so this ceiling is
# raised from the original 2GB to give it headroom while still catching a
# mistaken swap to a truly huge category (Books, Clothing, Home_and_Kitchen
# are each another order of magnitude beyond Video_Games).
MAX_EXPECTED_GZ_BYTES = 4 * 1024 ** 3  # 4GB


def download_via_hub():
    from huggingface_hub import hf_hub_download
    print("Downloading via huggingface_hub ...")
    review_path = hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=HF_REVIEW_PATH)
    meta_path = hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=HF_META_PATH)
    shutil.copy(review_path, REVIEW_GZ)
    shutil.copy(meta_path, META_GZ)


def download_via_url(url, dest):
    print(f"Downloading {url} -> {dest} ...")
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
        content_length = resp.getheader("Content-Length")
        if content_length and int(content_length) > MAX_EXPECTED_GZ_BYTES:
            raise SystemExit(
                f"Refusing to download: {url} reports {int(content_length):,} bytes, "
                f"over the {MAX_EXPECTED_GZ_BYTES:,}-byte sanity ceiling. "
                f"Check CATEGORY is really the small category you intended.")
        shutil.copyfileobj(resp, f, length=1024 * 1024)
    print(f"  done in {time.time()-t0:.0f}s")


def gunzip(src, dest):
    print(f"Decompressing {src} -> {dest} ...")
    with gzip.open(src, "rb") as fin, open(dest, "wb") as fout:
        shutil.copyfileobj(fin, fout)


def count_lines(path):
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for _ in f:
            n += 1
    return n


def main():
    if not (os.path.exists(REVIEW_GZ) and os.path.exists(META_GZ)):
        try:
            download_via_hub()
        except Exception as e:
            # Broad catch, not just ImportError: the HF hub copy of this dataset
            # is stored as plain .jsonl (no .gz), so a hub lookup for the .gz
            # filename this script expects 404s for every category (confirmed
            # against the HF file listing) — always falls through to the direct
            # URL below, which does serve genuine .jsonl.gz per the official
            # per-category table at https://amazon-reviews-2023.github.io/main.html.
            print(f"huggingface_hub download failed ({type(e).__name__}: {e}); "
                  "falling back to direct URL download.")
            download_via_url(REVIEW_URL, REVIEW_GZ)
            download_via_url(META_URL, META_GZ)
    else:
        print("Compressed files already present, skipping download.")

    for gz in [REVIEW_GZ, META_GZ]:
        size = os.path.getsize(gz)
        if size > MAX_EXPECTED_GZ_BYTES:
            raise SystemExit(
                f"{gz} is {size:,} bytes, over the {MAX_EXPECTED_GZ_BYTES:,}-byte sanity "
                f"ceiling — delete it and double check CATEGORY before retrying.")

    for gz, jsonl in [(REVIEW_GZ, REVIEW_JSONL), (META_GZ, META_JSONL)]:
        if not os.path.exists(jsonl):
            gunzip(gz, jsonl)

    print("\n" + "=" * 60)
    print(f"CATEGORY = {CATEGORY}")
    print("=" * 60)
    for label, gz, jsonl in [("review", REVIEW_GZ, REVIEW_JSONL), ("meta", META_GZ, META_JSONL)]:
        gz_mb = os.path.getsize(gz) / (1024 ** 2)
        jsonl_mb = os.path.getsize(jsonl) / (1024 ** 2)
        n_lines = count_lines(jsonl)
        print(f"{label:>7}: {gz_mb:6.1f}MB compressed, {jsonl_mb:7.1f}MB raw, {n_lines:,} lines")

    print("\nSanity check: review-file line count should be in the same order of "
          "magnitude as the #Rating column for this category in the official "
          "per-category table (https://amazon-reviews-2023.github.io/main.html). "
          "If it's off by 10x or more, double check CATEGORY.")


if __name__ == "__main__":
    main()

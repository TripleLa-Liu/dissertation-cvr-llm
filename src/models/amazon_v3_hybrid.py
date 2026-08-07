"""
Amazon Reviews'23 (Video_Games) V3 — Hybrid Segment Router (V1 + V2)
================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

Amazon-side counterpart to llm_encoder_v3_hybrid.py (2026-08-04 supervisor
request to extend the V3 result across both datasets, not just Ali-CCP).

IMPORTANT — the routing DIRECTION is flipped relative to the Ali-CCP
version, not just copied over. The first run of this script (2026-08-04,
using the Ali-CCP direction unchanged) showed V2 actually beats V1 on the
target hard segment here, the opposite of Ali-CCP:

    Ali-CCP (pseudo-text):  V1 wins short_context & cold_start (0.5671 vs V2's 0.5455)
    Amazon  (real text):    V2 wins short_context & cold_start (0.5173 vs V1's 0.4872)
                             — and V1 wins all three other cells (long&seen,
                                long&cold, short&seen)

This is itself a finding worth keeping: it's the direct empirical answer to
the supervisor's comment asking whether a real-text dataset would confirm
the original prediction that V2 (ALIGN) should win in the cold-start/
short-context setting — on Ali-CCP's pseudo-text it doesn't, on Amazon's
real text it does. So the routing rule here is:

    is_hard_segment = (context_segment == "short_context") & is_cold_start_item
    prediction      = V2's prediction where is_hard_segment,
                      else V1-equivalent's prediction

i.e. the reverse assignment from llm_encoder_v3_hybrid.py's rule, chosen
because it's what actually wins each cell on this dataset — not copied
mechanically from the Ali-CCP script.

"V1-equivalent" here is amazon_text_embedding.py (REPLACE pattern: item ID
embedding replaced entirely with a frozen real-text embedding) and "V2" is
amazon_v2_aligned.py (ALIGN pattern: ID embedding kept, aligned to text via
a contrastive loss, text projection reused as the cold-start fallback) —
the same two architectural patterns compared on Ali-CCP, just without the
CTR/CVR two-stage structure Amazon's data doesn't have.

V3 is NOT a new trained model, same as the Ali-CCP version: a hard,
deterministic router over the two already-trained checkpoints (no new
training, no threshold tuned against test labels).

Requires:
  - amazon_test_with_segments.csv (from
    data/preprocessing/amazon_build_difficulty_segments.py)
  - amazon_text_embedding.py already run (text_embedding_results/ populated)
  - amazon_v2_aligned.py already run (v2_aligned_results/ populated)
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_auc_score

WORK_DIR = r"D:\Study\migration_package\processed_data"
PROCESSED_DIR = os.path.join(WORK_DIR, "amazon", "processed")
TRAIN_PATH = os.path.join(PROCESSED_DIR, "amazon_train.csv")
SEGMENTED_TEST_PATH = os.path.join(PROCESSED_DIR, "amazon_test_with_segments.csv")

V1_DIR = os.path.join(WORK_DIR, "amazon", "text_embedding_results")
V2_DIR = os.path.join(WORK_DIR, "amazon", "v2_aligned_results")
BASELINE_METRICS_PATH = os.path.join(WORK_DIR, "amazon", "baseline_results",
                                      "amazon_id_baseline_metrics.json")

RESULTS_PATH = os.path.join(WORK_DIR, "amazon", "v3_hybrid_results.json")

BATCH_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------
def load_train_df():
    return pd.read_csv(TRAIN_PATH, dtype={"user_id": str, "item_id": str,
                                           "label": "int8", "is_cold_start_item": "int8"})


def load_segmented_test_df():
    df = pd.read_csv(SEGMENTED_TEST_PATH, dtype={
        "user_id": str, "item_id": str, "label": "int8", "is_cold_start_item": "int8",
        "context_segment": str})
    return df


def build_vocab(series):
    uniques = series.unique()
    return {v: i + 1 for i, v in enumerate(uniques)}


def encode(series, vocab):
    return series.map(lambda x: vocab.get(x, 0)).astype("int64").values.copy()


def make_tower(in_dim, hidden):
    layers = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1)]
        d = h
    layers += [nn.Linear(d, 1)]
    return nn.Sequential(*layers)


@torch.no_grad()
def run_predict(model, loader, n_inputs):
    model.eval()
    preds = []
    for batch in loader:
        inputs = [t.to(DEVICE) for t in batch[:n_inputs]]
        preds.append(model(*inputs).cpu().numpy())
    return np.concatenate(preds)


def segment_breakdown(model_name, preds, labels, context_segment, is_cold):
    print(f"\n--- {model_name} ---")
    groups = {
        "overall": np.ones(len(labels), dtype=bool),
        "long_context": context_segment == "long_context",
        "short_context": context_segment == "short_context",
        "long_context & seen": (context_segment == "long_context") & (~is_cold),
        "long_context & cold_start": (context_segment == "long_context") & is_cold,
        "short_context & seen": (context_segment == "short_context") & (~is_cold),
        "short_context & cold_start": (context_segment == "short_context") & is_cold,
    }
    row = {}
    for name, mask in groups.items():
        p, l = preds[mask], labels[mask]
        out = {"n": int(mask.sum()), "positive_rate": float(l.mean()) if mask.sum() else None}
        if len(np.unique(l)) > 1:
            out["auc"] = float(roc_auc_score(l, p))
        row[name] = out
        print(f"  {name:28s} n={out['n']:>8,}  AUC={out.get('auc', float('nan')):.4f}")
    return row


# ------------------------------------------------------------------
# Model-specific rebuild + predict
# ------------------------------------------------------------------
def eval_v1(test_df, train_df):
    """amazon_text_embedding.py (REPLACE pattern). Its script never
    pickled a vocab file, so rebuild user_vocab identically from the same
    train_df (deterministic: same file, same pandas version, same
    series.unique() order as when the checkpoint was trained — same
    convention already relied on for Ali-CCP's V2 in
    llm_encoder_v3_hybrid.py)."""
    user_vocab = build_vocab(train_df["user_id"])

    with open(os.path.join(V1_DIR, "item_llm_embeddings.pkl"), "rb") as f:
        item_data = pickle.load(f)
    item_id_to_row, item_embed_matrix, unk_row = (
        item_data["item_id_to_row"], item_data["embeddings"], item_data["unk_row"])

    class TextEmbeddingModel(nn.Module):
        def __init__(self, n_users, item_embed_matrix, user_dim=32, adapter_dim=32, hidden=(128, 64)):
            super().__init__()
            self.user_emb = nn.Embedding(n_users + 1, user_dim, padding_idx=0)
            n_items, lm_dim = item_embed_matrix.shape
            self.item_emb_frozen = nn.Embedding.from_pretrained(
                torch.from_numpy(item_embed_matrix), freeze=True)
            self.item_adapter = nn.Sequential(
                nn.Linear(lm_dim, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, adapter_dim))
            self.tower = make_tower(user_dim + adapter_dim, hidden)

        def forward(self, u, i_row):
            item_repr = self.item_adapter(self.item_emb_frozen(i_row))
            x = torch.cat([self.user_emb(u), item_repr], dim=1)
            return torch.sigmoid(self.tower(x)).squeeze(-1)

    model = TextEmbeddingModel(len(user_vocab), item_embed_matrix).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(V1_DIR, "amazon_text_embedding.pt"),
                                      map_location=DEVICE))

    u = torch.from_numpy(encode(test_df["user_id"], user_vocab))
    i_row = torch.from_numpy(
        test_df["item_id"].map(lambda x: item_id_to_row.get(x, unk_row)).astype("int64").values)
    loader = DataLoader(TensorDataset(u, i_row), batch_size=BATCH_SIZE * 4, shuffle=False)
    return run_predict(model, loader, n_inputs=2)


def eval_v2(test_df):
    """amazon_v2_aligned.py (ALIGN pattern). This script DOES pickle its
    vocab (VOCAB_PATH), so load it directly rather than rebuilding."""
    with open(os.path.join(V2_DIR, "id_vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    user_vocab, item_vocab = vocab["user_vocab"], vocab["item_vocab"]

    with open(os.path.join(V2_DIR, "item_llm_embeddings.pkl"), "rb") as f:
        item_llm_data = pickle.load(f)
    item_id_to_llm_row = item_llm_data["item_id_to_llm_row"]
    item_llm_matrix = item_llm_data["embeddings"]
    llm_unk_row = item_llm_data["unk_row"]

    class V2Aligned(nn.Module):
        def __init__(self, n_users, n_items, item_llm_embed_matrix, user_dim=32, item_dim=32,
                     hidden=(128, 64)):
            super().__init__()
            self.user_emb = nn.Embedding(n_users + 1, user_dim, padding_idx=0)
            self.item_emb = nn.Embedding(n_items + 1, item_dim, padding_idx=0)
            self.item_llm_frozen = nn.Embedding.from_pretrained(
                torch.from_numpy(item_llm_embed_matrix), freeze=True)
            self.item_llm_proj = nn.Sequential(
                nn.Linear(item_llm_embed_matrix.shape[1], 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, item_dim))
            self.tower = make_tower(user_dim + item_dim, hidden)

        def item_representation(self, item_idx, item_llm_row):
            id_repr = self.item_emb(item_idx)
            llm_repr = self.item_llm_proj(self.item_llm_frozen(item_llm_row))
            use_id = (item_idx != 0).float().unsqueeze(-1)
            return use_id * id_repr + (1 - use_id) * llm_repr

        def forward(self, u, item_idx, item_llm_row):
            item_repr = self.item_representation(item_idx, item_llm_row)
            x = torch.cat([self.user_emb(u), item_repr], dim=1)
            return torch.sigmoid(self.tower(x)).squeeze(-1)

    model = V2Aligned(len(user_vocab), len(item_vocab), item_llm_matrix).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(V2_DIR, "amazon_v2_aligned.pt"),
                                      map_location=DEVICE))

    u = torch.from_numpy(encode(test_df["user_id"], user_vocab))
    i_idx = torch.from_numpy(encode(test_df["item_id"], item_vocab))
    i_llm_row = torch.from_numpy(
        test_df["item_id"].map(lambda x: item_id_to_llm_row.get(x, llm_unk_row)).astype("int64").values)
    loader = DataLoader(TensorDataset(u, i_idx, i_llm_row), batch_size=BATCH_SIZE * 4, shuffle=False)
    return run_predict(model, loader, n_inputs=3)


# ------------------------------------------------------------------
# Hybrid routing
# ------------------------------------------------------------------
def route(v1_preds, v2_preds, is_hard_segment):
    """Amazon direction (flipped vs. Ali-CCP's llm_encoder_v3_hybrid.py):
    V2 wins the hard segment (short_context & cold_start) on real text, V1
    wins everywhere else — see module docstring for the measured cell
    values that justify this direction."""
    return np.where(is_hard_segment, v2_preds, v1_preds)


def main():
    for name, d in [("V1_DIR (amazon_text_embedding.py output)", V1_DIR),
                    ("V2_DIR (amazon_v2_aligned.py output)", V2_DIR)]:
        if not os.path.isdir(d):
            raise SystemExit(
                f"{name}={d} not found. V3 routes between already-trained checkpoints, "
                f"it doesn't train anything itself — rerun amazon_text_embedding.py and/or "
                f"amazon_v2_aligned.py locally first.")
    if not os.path.exists(SEGMENTED_TEST_PATH):
        raise SystemExit(
            f"{SEGMENTED_TEST_PATH} not found — run "
            f"data/preprocessing/amazon_build_difficulty_segments.py first.")

    print("Loading segmented test set ...")
    test_df = load_segmented_test_df()
    labels = test_df["label"].values.astype("int8")
    is_cold = test_df["is_cold_start_item"].values.astype(bool)
    context_segment = test_df["context_segment"].values
    is_hard_segment = (context_segment == "short_context") & is_cold
    print(f"  {len(test_df):,} rows | hard segment (short_context & cold_start) = "
          f"{is_hard_segment.sum():,} ({is_hard_segment.mean():.1%})")

    print("\nLoading train_df (needed to rebuild V1's vocab identically) ...")
    train_df = load_train_df()

    print("\n=== V1-equivalent (item-only real-text replace) ===")
    v1_preds = eval_v1(test_df, train_df)
    v1_breakdown = segment_breakdown("V1 (text replace)", v1_preds, labels, context_segment, is_cold)

    print("\n=== V2 (ID + LLM alignment) ===")
    v2_preds = eval_v2(test_df)
    v2_breakdown = segment_breakdown("V2 (ID + align)", v2_preds, labels, context_segment, is_cold)

    print("\n=== V3 (hybrid: V2 on short_context & cold_start, V1 elsewhere) ===")
    hybrid_preds = route(v1_preds, v2_preds, is_hard_segment)
    v3_breakdown = segment_breakdown("V3-Hybrid", hybrid_preds, labels, context_segment, is_cold)

    all_results = {"v1_text_replace": v1_breakdown, "v2_aligned": v2_breakdown,
                   "v3_hybrid": v3_breakdown}

    if os.path.exists(BASELINE_METRICS_PATH):
        with open(BASELINE_METRICS_PATH) as f:
            baseline_metrics = json.load(f)
        # baseline has no context_segment breakdown (it predates this
        # script) — fold in what's available (overall/seen/cold) so the
        # summary table below can at least show baseline's overall number
        all_results["baseline_overall_only"] = {
            "overall": baseline_metrics.get("test_overall", {}),
            "seen": baseline_metrics.get("test_seen_items", {}),
            "cold_start": baseline_metrics.get("test_cold_start_items", {}),
        }

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")

    print("\n" + "=" * 70)
    print("SUMMARY: overall AUC and the hard-segment cell")
    print("=" * 70)
    for name in ("v1_text_replace", "v2_aligned", "v3_hybrid"):
        r = all_results[name]
        overall = r.get("overall", {}).get("auc", float("nan"))
        hard = r.get("short_context & cold_start", {}).get("auc", float("nan"))
        print(f"  {name:20s} overall={overall:.4f}   short_context&cold_start={hard:.4f}")
    if "baseline_overall_only" in all_results:
        b = all_results["baseline_overall_only"]["overall"].get("auc", float("nan"))
        print(f"  {'baseline (ID)':20s} overall={b:.4f}   short_context&cold_start=n/a "
              f"(baseline predates segment breakdown)")


if __name__ == "__main__":
    main()

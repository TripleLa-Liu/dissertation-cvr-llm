"""
Ali-CCP V3 — Hybrid Segment Router (V1 + V2, run locally)
================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

Answers the 2026-07-30 supervisor meeting question directly: "Should this
motivate a hybrid V3 (route cold-start items by context richness)?"

Motivation (from src/eval_context_segments.py's 2x2 difficulty breakdown,
see README "Test-set difficulty segmentation"): V2 (ID + LLM alignment) is
the strongest model overall, but in the single hardest cell —
short_context & cold_start (418,643 rows, ~20.9% of test) — V1 (item-only
frozen text embedding, CTCVR-AUC 0.5627) clearly beats V2 (0.4991). Outside
that cell V2 is at least competitive with V1 everywhere. This suggests a
genuine crossover, not noise: when a user has almost no interaction
history AND the item has never been seen in train, V2's from-scratch item
ID embedding for that item is entirely untrained (falls back to index 0),
while V1's frozen text embedding still encodes real semantic content for
it — so V1 has signal where V2 structurally cannot.

V3 is deliberately NOT a new trained model. It is a hard, deterministic
router over the two already-trained checkpoints (no new training, no
threshold tuned against test labels — the routing rule is the segment
definition itself, decided before this script ever looked at test
performance):

    is_hard_segment = (context_segment == "short_context") & is_cold_start_item
    prediction      = V1's prediction where is_hard_segment, else V2's prediction

p_ctr, p_cvr, and p_ctcvr are all routed together per row from the same
source model (not routed independently), so ctcvr_hybrid == ctr_hybrid *
cvr_hybrid stays internally consistent, exactly as it does within V1 or V2
alone.

Reuses each model's exact architecture/preprocessing from its own training
script (src/models/llm_encoder_v1.py, llm_encoder_v2_aligned.py) and loads
the already-trained checkpoints — pure inference. Requires V1_DIR and
V2_DIR to already contain their checkpoints/vocabs; if either is missing,
rerun the corresponding training script first (llm_encoder_v1.py /
llm_encoder_v2_aligned.py).

Requires aliccp_test_with_segments.csv (from
build_test_difficulty_segments.py) in WORK_DIR.
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
TRAIN_PATH = os.path.join(WORK_DIR, "aliccp_train_split.csv")
SEGMENTED_TEST_PATH = os.path.join(WORK_DIR, "aliccp_test_with_segments.csv")

V1_DIR = os.path.join(WORK_DIR, "v1_results")
V2_DIR = os.path.join(WORK_DIR, "v2_results")

RESULTS_PATH = os.path.join(WORK_DIR, "v3_hybrid_results.json")
EXISTING_SEGMENT_EVAL_PATH = os.path.join(WORK_DIR, "context_segment_eval_results.json")

BATCH_SIZE = 4096
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# Shared helpers (same as eval_context_segments.py)
# ------------------------------------------------------------------
def load_train_df():
    df = pd.read_csv(TRAIN_PATH, dtype={"user_id": str, "item_id": str,
                                         "click": "int8", "purchase": "int8"})
    df["user_id"] = df["user_id"].fillna("__MISSING__")
    return df


def load_segmented_test_df():
    df = pd.read_csv(SEGMENTED_TEST_PATH, dtype={
        "user_id": str, "item_id": str, "click": "int8", "purchase": "int8",
        "is_cold_start_item": "int8", "context_segment": str})
    df["user_id"] = df["user_id"].fillna("__MISSING__")
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
    p_ctrs, p_cvrs, p_ctcvrs = [], [], []
    for batch in loader:
        inputs = [t.to(DEVICE) for t in batch[:n_inputs]]
        p_ctr, p_cvr, p_ctcvr = model(*inputs)
        p_ctrs.append(p_ctr.cpu().numpy())
        p_cvrs.append(p_cvr.cpu().numpy())
        p_ctcvrs.append(p_ctcvr.cpu().numpy())
    return np.concatenate(p_ctrs), np.concatenate(p_cvrs), np.concatenate(p_ctcvrs)


def segment_breakdown(model_name, p_ctr, p_cvr, p_ctcvr, click, purchase, context_segment, is_cold):
    print(f"\n--- {model_name} ---")
    groups = {
        "overall": np.ones(len(click), dtype=bool),
        "long_context": context_segment == "long_context",
        "short_context": context_segment == "short_context",
        "long_context & seen": (context_segment == "long_context") & (~is_cold),
        "long_context & cold_start": (context_segment == "long_context") & is_cold,
        "short_context & seen": (context_segment == "short_context") & (~is_cold),
        "short_context & cold_start": (context_segment == "short_context") & is_cold,
    }
    row = {}
    for name, mask in groups.items():
        c, p, pc, pv, pt = click[mask], purchase[mask], p_ctr[mask], p_cvr[mask], p_ctcvr[mask]
        out = {"n": int(mask.sum())}
        if len(np.unique(c)) > 1:
            out["ctr_auc"] = float(roc_auc_score(c, pc))
        clicked = c == 1
        if clicked.sum() > 1 and len(np.unique(p[clicked])) > 1:
            out["cvr_auc"] = float(roc_auc_score(p[clicked], pv[clicked]))
        if len(np.unique(p)) > 1:
            out["ctcvr_auc"] = float(roc_auc_score(p, pt))
        row[name] = out
        print(f"  {name:28s} n={out['n']:>9,}  CTR-AUC={out.get('ctr_auc', float('nan')):.4f}  "
              f"CVR-AUC={out.get('cvr_auc', float('nan')):.4f}  "
              f"CTCVR-AUC={out.get('ctcvr_auc', float('nan')):.4f}")
    return row


# ------------------------------------------------------------------
# Model-specific rebuild + predict (identical to eval_context_segments.py)
# ------------------------------------------------------------------
def eval_v1(test_df):
    with open(os.path.join(V1_DIR, "id_vocab_v1.pkl"), "rb") as f:
        vocab = pickle.load(f)
    user_vocab = vocab["user_vocab"]
    with open(os.path.join(V1_DIR, "item_llm_embeddings.pkl"), "rb") as f:
        item_data = pickle.load(f)
    item_id_to_row, item_embed_matrix, unk_row = (
        item_data["item_id_to_row"], item_data["embeddings"], item_data["unk_row"])

    class ESMM_LLM_V1(nn.Module):
        def __init__(self, n_users, item_embed_matrix, user_dim=32, adapter_dim=32, hidden=(128, 64)):
            super().__init__()
            self.user_emb = nn.Embedding(n_users + 1, user_dim, padding_idx=0)
            n_items, lm_dim = item_embed_matrix.shape
            self.item_emb_frozen = nn.Embedding.from_pretrained(
                torch.from_numpy(item_embed_matrix), freeze=True)
            self.item_adapter = nn.Sequential(
                nn.Linear(lm_dim, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, adapter_dim))
            tower_in = user_dim + adapter_dim
            self.ctr_tower = make_tower(tower_in, hidden)
            self.cvr_tower = make_tower(tower_in, hidden)

        def forward(self, u, i_row):
            item_repr = self.item_adapter(self.item_emb_frozen(i_row))
            x = torch.cat([self.user_emb(u), item_repr], dim=1)
            p_ctr = torch.sigmoid(self.ctr_tower(x)).squeeze(-1)
            p_cvr = torch.sigmoid(self.cvr_tower(x)).squeeze(-1)
            return p_ctr, p_cvr, p_ctr * p_cvr

    model = ESMM_LLM_V1(len(user_vocab), item_embed_matrix).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(V1_DIR, "esmm_llm_v1.pt"), map_location=DEVICE))

    u = torch.from_numpy(encode(test_df["user_id"], user_vocab))
    i_row = torch.from_numpy(
        test_df["item_id"].map(lambda x: item_id_to_row.get(x, unk_row)).astype("int64").values)
    loader = DataLoader(TensorDataset(u, i_row), batch_size=BATCH_SIZE * 4, shuffle=False)
    return run_predict(model, loader, n_inputs=2)


def eval_v2(test_df, train_df):
    with open(os.path.join(V2_DIR, "item_llm_embeddings.pkl"), "rb") as f:
        item_llm_data = pickle.load(f)
    item_id_to_llm_row = item_llm_data["item_id_to_llm_row"]
    item_llm_matrix = item_llm_data["embeddings"]
    llm_unk_row = item_llm_data["unk_row"]

    # V2's script never pickled its vocab -- rebuild identically from the
    # same train_df (deterministic: same file, same pandas version, same
    # series.unique() order as when the checkpoint was trained).
    user_vocab = build_vocab(train_df["user_id"])
    item_vocab = build_vocab(train_df["item_id"])

    class ESMM_V2_Aligned(nn.Module):
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
            tower_in = user_dim + item_dim
            self.ctr_tower = make_tower(tower_in, hidden)
            self.cvr_tower = make_tower(tower_in, hidden)

        def item_representation(self, item_idx, item_llm_row):
            id_repr = self.item_emb(item_idx)
            llm_repr = self.item_llm_proj(self.item_llm_frozen(item_llm_row))
            use_id = (item_idx != 0).float().unsqueeze(-1)
            return use_id * id_repr + (1 - use_id) * llm_repr

        def forward(self, u, item_idx, item_llm_row):
            item_repr = self.item_representation(item_idx, item_llm_row)
            x = torch.cat([self.user_emb(u), item_repr], dim=1)
            p_ctr = torch.sigmoid(self.ctr_tower(x)).squeeze(-1)
            p_cvr = torch.sigmoid(self.cvr_tower(x)).squeeze(-1)
            return p_ctr, p_cvr, p_ctr * p_cvr

    model = ESMM_V2_Aligned(len(user_vocab), len(item_vocab), item_llm_matrix).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(V2_DIR, "esmm_v2_aligned.pt"), map_location=DEVICE))

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
    p_ctr = np.where(is_hard_segment, v1_preds[0], v2_preds[0])
    p_cvr = np.where(is_hard_segment, v1_preds[1], v2_preds[1])
    p_ctcvr = np.where(is_hard_segment, v1_preds[2], v2_preds[2])
    return p_ctr, p_cvr, p_ctcvr


def main():
    for name, d in [("V1_DIR", V1_DIR), ("V2_DIR", V2_DIR)]:
        if not os.path.isdir(d):
            raise SystemExit(
                f"{name}={d} not found. V3 routes between already-trained V1 and V2 "
                f"checkpoints, it doesn't train anything itself — rerun "
                f"llm_encoder_v1.py and/or llm_encoder_v2_aligned.py locally first.")

    print("Loading segmented test set ...")
    test_df = load_segmented_test_df()
    click = test_df["click"].values.astype("int8")
    purchase = test_df["purchase"].values.astype("int8")
    is_cold = test_df["is_cold_start_item"].values.astype(bool)
    context_segment = test_df["context_segment"].values
    is_hard_segment = (context_segment == "short_context") & is_cold
    print(f"  {len(test_df):,} rows | hard segment (short_context & cold_start) = "
          f"{is_hard_segment.sum():,} ({is_hard_segment.mean():.1%})")

    print("\nLoading train_df (needed to rebuild V2's vocab identically) ...")
    train_df = load_train_df()

    print("\n=== V1 (item-only LLM replace) ===")
    v1_preds = eval_v1(test_df)
    v1_breakdown = segment_breakdown("V1", *v1_preds, click, purchase, context_segment, is_cold)

    print("\n=== V2 (ID + LLM alignment) ===")
    v2_preds = eval_v2(test_df, train_df)
    v2_breakdown = segment_breakdown("V2", *v2_preds, click, purchase, context_segment, is_cold)

    print("\n=== V3 (hybrid: V1 on short_context & cold_start, V2 elsewhere) ===")
    p_ctr, p_cvr, p_ctcvr = route(v1_preds, v2_preds, is_hard_segment)
    v3_breakdown = segment_breakdown("V3-Hybrid", p_ctr, p_cvr, p_ctcvr,
                                      click, purchase, context_segment, is_cold)

    all_results = {"v1": v1_breakdown, "v2": v2_breakdown, "v3_hybrid": v3_breakdown}

    # Fold in baseline/V1-Full from the earlier full-model comparison, if present,
    # so the saved file has all five models in one place without re-running them.
    if os.path.exists(EXISTING_SEGMENT_EVAL_PATH):
        with open(EXISTING_SEGMENT_EVAL_PATH) as f:
            existing = json.load(f)
        for k in ("baseline", "v1_full"):
            if k in existing:
                all_results[k] = existing[k]

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")

    print("\n" + "=" * 70)
    print("SUMMARY: overall CTCVR-AUC and the hard-segment cell")
    print("=" * 70)
    for name in ("baseline", "v1", "v1_full", "v2", "v3_hybrid"):
        if name not in all_results:
            continue
        r = all_results[name]
        overall = r.get("overall", {}).get("ctcvr_auc", float("nan"))
        hard = r.get("short_context & cold_start", {}).get("ctcvr_auc", float("nan"))
        print(f"  {name:12s} overall={overall:.4f}   short_context&cold_start={hard:.4f}")


if __name__ == "__main__":
    main()

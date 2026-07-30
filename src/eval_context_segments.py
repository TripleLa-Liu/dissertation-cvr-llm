"""
Ali-CCP test-set difficulty-segment evaluation (run locally)
================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

WHAT THIS DOES
---------------
Completes the meeting notes' "test set design" item: breaks down all four
already-trained models' (Baseline / V1 / V1-Full / V2) test performance by
the new context_segment column (long_context / short_context, added by
data/preprocessing/build_test_difficulty_segments.py -- see README "Test
set design" section for the "session interaction count" proxy rationale)
CROSSED with the existing is_cold_start_item flag, giving a 2x2 difficulty
matrix instead of just the single cold-start split reported so far.

Reuses each model's EXACT architecture/preprocessing from its own training
script and loads the already-trained checkpoint (no retraining) -- this
is pure inference, so it should run in well under a minute even on CPU.

REQUIRES: aliccp_test_with_segments.csv already exists in WORK_DIR (built
2026-07-21, already in your _processed folder -- no action needed, just
confirm the file is there before running).

USAGE
-----
python eval_context_segments.py
Send me the printed table.
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

WORK_DIR = r"E:\BaiduNetdiskDownload\Dataset\_processed"
TRAIN_PATH = os.path.join(WORK_DIR, "aliccp_train_split.csv")
SEGMENTED_TEST_PATH = os.path.join(WORK_DIR, "aliccp_test_with_segments.csv")

BASELINE_DIR = os.path.join(WORK_DIR, "baseline_results")
V1_DIR = os.path.join(WORK_DIR, "v1_results")
V1_FULL_DIR = os.path.join(WORK_DIR, "v1_full_results")
V2_DIR = os.path.join(WORK_DIR, "v2_results")

BATCH_SIZE = 4096
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# Shared helpers
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
    return series.map(lambda x: vocab.get(x, 0)).astype("int64").values


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
# Model-specific rebuild + predict
# ------------------------------------------------------------------
def eval_baseline(test_df):
    with open(os.path.join(BASELINE_DIR, "id_vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    user_vocab, item_vocab = vocab["user_vocab"], vocab["item_vocab"]

    class ESMM(nn.Module):
        def __init__(self, n_users, n_items, embed_dim=32, hidden=(128, 64)):
            super().__init__()
            self.user_emb = nn.Embedding(n_users + 1, embed_dim, padding_idx=0)
            self.item_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
            self.ctr_tower = make_tower(embed_dim * 2, hidden)
            self.cvr_tower = make_tower(embed_dim * 2, hidden)

        def forward(self, u, i):
            x = torch.cat([self.user_emb(u), self.item_emb(i)], dim=1)
            p_ctr = torch.sigmoid(self.ctr_tower(x)).squeeze(-1)
            p_cvr = torch.sigmoid(self.cvr_tower(x)).squeeze(-1)
            return p_ctr, p_cvr, p_ctr * p_cvr

    model = ESMM(len(user_vocab), len(item_vocab)).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(BASELINE_DIR, "esmm_id_baseline.pt"),
                                      map_location=DEVICE))

    u = torch.from_numpy(encode(test_df["user_id"], user_vocab))
    i = torch.from_numpy(encode(test_df["item_id"], item_vocab))
    loader = DataLoader(TensorDataset(u, i), batch_size=BATCH_SIZE * 4, shuffle=False)
    return run_predict(model, loader, n_inputs=2)


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


def eval_v1_full(test_df):
    with open(os.path.join(V1_FULL_DIR, "item_llm_embeddings.pkl"), "rb") as f:
        item_data = pickle.load(f)
    with open(os.path.join(V1_FULL_DIR, "user_llm_embeddings.pkl"), "rb") as f:
        user_data = pickle.load(f)

    class ESMM_LLM_V1_Full(nn.Module):
        def __init__(self, user_embed_matrix, item_embed_matrix, adapter_dim=32, hidden=(128, 64)):
            super().__init__()
            self.user_emb_frozen = nn.Embedding.from_pretrained(
                torch.from_numpy(user_embed_matrix), freeze=True)
            self.item_emb_frozen = nn.Embedding.from_pretrained(
                torch.from_numpy(item_embed_matrix), freeze=True)
            self.user_adapter = nn.Sequential(
                nn.Linear(user_embed_matrix.shape[1], 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, adapter_dim))
            self.item_adapter = nn.Sequential(
                nn.Linear(item_embed_matrix.shape[1], 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, adapter_dim))
            tower_in = adapter_dim * 2
            self.ctr_tower = make_tower(tower_in, hidden)
            self.cvr_tower = make_tower(tower_in, hidden)

        def forward(self, u_row, i_row):
            user_repr = self.user_adapter(self.user_emb_frozen(u_row))
            item_repr = self.item_adapter(self.item_emb_frozen(i_row))
            x = torch.cat([user_repr, item_repr], dim=1)
            p_ctr = torch.sigmoid(self.ctr_tower(x)).squeeze(-1)
            p_cvr = torch.sigmoid(self.cvr_tower(x)).squeeze(-1)
            return p_ctr, p_cvr, p_ctr * p_cvr

    model = ESMM_LLM_V1_Full(user_data["embeddings"], item_data["embeddings"]).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(V1_FULL_DIR, "esmm_llm_v1_full.pt"),
                                      map_location=DEVICE))

    i_row = torch.from_numpy(test_df["item_id"].map(
        lambda x: item_data["id_to_row"].get(x, item_data["unk_row"])).astype("int64").values)
    u_row = torch.from_numpy(test_df["user_id"].map(
        lambda x: user_data["id_to_row"].get(x, user_data["unk_row"])).astype("int64").values)
    loader = DataLoader(TensorDataset(u_row, i_row), batch_size=BATCH_SIZE * 4, shuffle=False)
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
def main():
    print("Loading segmented test set ...")
    test_df = load_segmented_test_df()
    click = test_df["click"].values.astype("int8")
    purchase = test_df["purchase"].values.astype("int8")
    is_cold = test_df["is_cold_start_item"].values.astype(bool)
    context_segment = test_df["context_segment"].values
    print(f"  {len(test_df):,} rows | "
          f"long_context={sum(context_segment=='long_context'):,} "
          f"short_context={sum(context_segment=='short_context'):,}")

    print("\nLoading train_df (needed to rebuild V2's vocab identically) ...")
    train_df = load_train_df()

    all_results = {}

    print("\n=== Baseline (ID embedding) ===")
    p_ctr, p_cvr, p_ctcvr = eval_baseline(test_df)
    all_results["baseline"] = segment_breakdown("Baseline", p_ctr, p_cvr, p_ctcvr,
                                                 click, purchase, context_segment, is_cold)

    print("\n=== V1 (item-only LLM replace) ===")
    p_ctr, p_cvr, p_ctcvr = eval_v1(test_df)
    all_results["v1"] = segment_breakdown("V1", p_ctr, p_cvr, p_ctcvr,
                                           click, purchase, context_segment, is_cold)

    print("\n=== V1-Full (item+user LLM replace) ===")
    p_ctr, p_cvr, p_ctcvr = eval_v1_full(test_df)
    all_results["v1_full"] = segment_breakdown("V1-Full", p_ctr, p_cvr, p_ctcvr,
                                                click, purchase, context_segment, is_cold)

    print("\n=== V2 (ID + LLM alignment) ===")
    p_ctr, p_cvr, p_ctcvr = eval_v2(test_df, train_df)
    all_results["v2"] = segment_breakdown("V2", p_ctr, p_cvr, p_ctcvr,
                                           click, purchase, context_segment, is_cold)

    out_path = os.path.join(WORK_DIR, "context_segment_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full breakdown to {out_path}")
    print("\nSend me the printed tables above (or the JSON file) -- I'll write the "
          "easy/hard segment comparison into README/Overleaf.")


if __name__ == "__main__":
    main()

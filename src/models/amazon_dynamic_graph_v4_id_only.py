"""
Amazon Reviews'23 (Video_Games) V4-ID-Only — Pure-ID Dynamic Graph Ablation
============================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

Ablation of amazon_dynamic_graph_v4.py: same temporal target-attention
history aggregator, but with the LLM/text branch removed entirely — item
representation is a plain learned ID embedding (identical convention to
amazon_id_baseline.py: index 0 = UNK for cold-start items), no
item_llm_proj, no frozen MiniLM embeddings, no alignment loss.

Purpose (2026-08-07, post V4 5-seed result): V4 (LLM+Graph combined) beat
Baseline and V1 on test_overall but not V2, and showed NO significant
cold-start advantage over any of Baseline/V1/V2 (see
results/multiseed_comparison_summary.md). That combined result can't tell
us whether the temporal-graph mechanism itself is pulling its weight, or
whether V4's edge over Baseline/V1 is coming entirely from the LLM-
alignment half (the part it shares with V2) with the graph aggregator
contributing nothing on top. This ablation isolates that: compare this
(ID + graph, no LLM) against amazon_id_baseline.py (ID, no graph, no LLM)
to test "does temporal graph structure help over the plain ID baseline",
with the LLM variable held out entirely.

Architecture:
  - user_emb, item_emb: learned nn.Embedding, identical to
    amazon_id_baseline.py (index 0 = UNK).
  - History items use the SAME item_emb table as the target item (no
    separate LLM lookup needed at all — this script has no dependency on
    amazon_item_text.csv or sentence-transformers).
  - TemporalAttention: identical mechanism to amazon_dynamic_graph_v4.py
    (target-attention + learned recency-decay bias from real Delta-t).
  - Final tower: concat(user_emb, dynamic_user_repr, target item_emb) ->
    MLP -> sigmoid. Same hidden dims as every other Amazon script for
    comparability.

Requires amazon_train.csv / amazon_val.csv / amazon_test.csv (with
timestamp column, 2026-08-07+) and amazon_user_histories.pkl (from
amazon_build_user_histories.py) — no amazon_item_text.csv needed.
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    raise SystemExit("Please install scikit-learn: pip install scikit-learn")

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
WORK_DIR = r"D:\Study\migration_package\processed_data"
PROCESSED_DIR = os.path.join(WORK_DIR, "amazon", "processed")
RESULTS_DIR = os.path.join(WORK_DIR, "amazon", "v4_id_only_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(PROCESSED_DIR, "amazon_train.csv")
VAL_PATH = os.path.join(PROCESSED_DIR, "amazon_val.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "amazon_test.csv")
HISTORIES_PATH = os.path.join(PROCESSED_DIR, "amazon_user_histories.pkl")

VOCAB_PATH = os.path.join(RESULTS_DIR, "id_vocab.pkl")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "amazon_dynamic_graph_v4_id_only.pt")
METRICS_PATH = os.path.join(RESULTS_DIR, "amazon_dynamic_graph_v4_id_only_metrics.json")

USER_EMBED_DIM = 32
ITEM_EMBED_DIM = 32
HIDDEN = [128, 64]
BATCH_SIZE = 1024
EPOCHS = 20
PATIENCE = 3
LR = 1e-3
WEIGHT_DECAY = 1e-6
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED, help="random seed (default: 42)")
    return p.parse_args()


def seed_suffixed(path, seed):
    if seed == SEED:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_seed{seed}{ext}"


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
def load_csv(path):
    return pd.read_csv(path, dtype={"user_id": str, "item_id": str,
                                     "label": "int8", "is_cold_start_item": "int8"})


def build_vocab(series):
    uniques = series.unique()
    return {v: i + 1 for i, v in enumerate(uniques)}


def encode(series, vocab):
    return series.map(lambda x: vocab.get(x, 0)).astype("int64").values.copy()


def make_target_tensors(df, user_vocab, item_vocab):
    u = encode(df["user_id"], user_vocab)
    i_idx = encode(df["item_id"], item_vocab)
    label = df["label"].values.astype("float32")
    return u, i_idx, label


# ------------------------------------------------------------------
# Causal history tensors — same causality logic as
# amazon_dynamic_graph_v4.py's compute_history_tensors, but no LLM row
# needed (history items are looked up in the same item_emb table as the
# target item, via item_vocab alone).
# ------------------------------------------------------------------
def compute_history_tensors(df, histories, item_vocab, max_hist):
    n = len(df)
    hist_idx = np.zeros((n, max_hist), dtype="int64")
    hist_delta_days = np.zeros((n, max_hist), dtype="float32")
    hist_mask = np.zeros((n, max_hist), dtype="bool")

    user_ids = df["user_id"].to_numpy()
    row_ts = df["timestamp"].to_numpy()
    empty_items = np.array([], dtype=object)
    empty_ts = np.array([], dtype="int64")

    n_with_history = 0
    for row_i in range(n):
        uid, t = user_ids[row_i], row_ts[row_i]
        items, ts = histories.get(uid, (empty_items, empty_ts))
        n_prior = int(np.searchsorted(ts, t, side="left"))
        if n_prior == 0:
            continue
        n_with_history += 1
        start = max(0, n_prior - max_hist)
        sel_items = items[start:n_prior]
        sel_ts = ts[start:n_prior]
        for j in range(len(sel_items)):
            hist_idx[row_i, j] = item_vocab.get(sel_items[j], 0)
            hist_delta_days[row_i, j] = max(0.0, (t - sel_ts[j]) / 86400.0)
            hist_mask[row_i, j] = True

    print(f"    {n_with_history:,}/{n:,} rows ({n_with_history/n:.1%}) have >=1 prior history item")
    return hist_idx, hist_delta_days, hist_mask


def make_tensors(df, user_vocab, item_vocab, histories, max_hist):
    u, i_idx, label = make_target_tensors(df, user_vocab, item_vocab)
    hist_idx, hist_delta_days, hist_mask = compute_history_tensors(df, histories, item_vocab, max_hist)
    return (torch.from_numpy(u), torch.from_numpy(i_idx),
            torch.from_numpy(hist_idx), torch.from_numpy(hist_delta_days), torch.from_numpy(hist_mask),
            torch.from_numpy(label))


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
class TemporalAttention(nn.Module):
    """Identical mechanism to amazon_dynamic_graph_v4.py's TemporalAttention
    — see that file for the design rationale."""

    def __init__(self):
        super().__init__()
        self.raw_decay = nn.Parameter(torch.tensor(0.1))

    def forward(self, query, keys, values, delta_days, mask):
        scores = torch.einsum("bd,bnd->bn", query, keys) / (query.size(-1) ** 0.5)
        decay = F.softplus(self.raw_decay)
        scores = scores - decay * torch.log1p(delta_days.clamp(min=0))
        scores = scores.masked_fill(~mask, float("-inf"))

        has_history = mask.any(dim=1)
        weights = torch.zeros_like(scores)
        if has_history.any():
            weights[has_history] = F.softmax(scores[has_history], dim=-1)
        out = torch.einsum("bn,bnd->bd", weights, values)
        return out


class DynamicGraphIDOnly(nn.Module):
    def __init__(self, n_users, n_items, user_dim=32, item_dim=32, hidden=(128, 64)):
        super().__init__()
        self.user_emb = nn.Embedding(n_users + 1, user_dim, padding_idx=0)
        self.item_emb = nn.Embedding(n_items + 1, item_dim, padding_idx=0)
        self.temporal_attn = TemporalAttention()

        d = user_dim + item_dim + item_dim  # user_emb + dynamic_user_repr + target item_emb
        layers = []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.tower = nn.Sequential(*layers)

    def forward(self, u, item_idx, hist_idx, hist_delta_days, hist_mask):
        target_repr = self.item_emb(item_idx)          # (B, item_dim)
        hist_repr = self.item_emb(hist_idx)             # (B, N, item_dim)
        dynamic_user_repr = self.temporal_attn(target_repr, hist_repr, hist_repr, hist_delta_days, hist_mask)
        x = torch.cat([self.user_emb(u), dynamic_user_repr, target_repr], dim=1)
        return torch.sigmoid(self.tower(x)).squeeze(-1)


# ------------------------------------------------------------------
# Train / eval
# ------------------------------------------------------------------
def train_epoch(model, loader, optimizer, bce):
    model.train()
    total_loss, n = 0.0, 0
    for u, item_idx, hist_idx, hist_delta_days, hist_mask, label in loader:
        u, item_idx = u.to(DEVICE), item_idx.to(DEVICE)
        hist_idx, hist_delta_days, hist_mask = hist_idx.to(DEVICE), hist_delta_days.to(DEVICE), hist_mask.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()
        p = model(u, item_idx, hist_idx, hist_delta_days, hist_mask)
        loss = bce(p, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(u)
        n += len(u)
    return total_loss / n


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, labels = [], []
    for u, item_idx, hist_idx, hist_delta_days, hist_mask, label in loader:
        u, item_idx = u.to(DEVICE), item_idx.to(DEVICE)
        hist_idx, hist_delta_days, hist_mask = hist_idx.to(DEVICE), hist_delta_days.to(DEVICE), hist_mask.to(DEVICE)
        p = model(u, item_idx, hist_idx, hist_delta_days, hist_mask)
        preds.append(p.cpu().numpy())
        labels.append(label.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def compute_metrics(preds, labels, mask=None):
    if mask is not None:
        preds, labels = preds[mask], labels[mask]
    out = {"n_rows": int(len(labels)), "positive_rate": float(labels.mean())}
    if len(np.unique(labels)) > 1:
        out["auc"] = float(roc_auc_score(labels, preds))
    return out


def main():
    args = parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    checkpoint_path = seed_suffixed(CHECKPOINT_PATH, seed)
    metrics_path = seed_suffixed(METRICS_PATH, seed)
    if seed != SEED:
        print(f"Running with --seed={seed} (non-default): outputs -> "
              f"{os.path.basename(checkpoint_path)}, {os.path.basename(metrics_path)}")

    if not os.path.exists(HISTORIES_PATH):
        raise SystemExit(
            f"{HISTORIES_PATH} not found — run "
            f"data/preprocessing/amazon_build_user_histories.py first.")

    print(f"Loading {HISTORIES_PATH} ...")
    with open(HISTORIES_PATH, "rb") as f:
        hist_data = pickle.load(f)
    histories = hist_data["histories"]
    max_hist = hist_data["max_history"]
    print(f"  {len(histories):,} users, max_history={max_hist}")

    print("\nLoading train/val/test CSVs ...")
    t0 = time.time()
    train_df = load_csv(TRAIN_PATH)
    val_df = load_csv(VAL_PATH)
    test_df = load_csv(TEST_PATH)
    if "timestamp" not in train_df.columns:
        raise SystemExit("amazon_train.csv has no `timestamp` column — rerun amazon_build_dataset.py.")
    print(f"  train={len(train_df):,} val={len(val_df):,} test={len(test_df):,} "
          f"({time.time()-t0:.0f}s)")

    print("Building user/item vocab from TRAIN only ...")
    user_vocab = build_vocab(train_df["user_id"])
    item_vocab = build_vocab(train_df["item_id"])
    n_users, n_items = len(user_vocab), len(item_vocab)
    print(f"  n_users={n_users:,} n_items={n_items:,}")
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump({"user_vocab": user_vocab, "item_vocab": item_vocab}, f)

    print("\nPrecomputing causal history tensors ...")
    t0 = time.time()
    print("  train:")
    train_t = make_tensors(train_df, user_vocab, item_vocab, histories, max_hist)
    print("  val:")
    val_t = make_tensors(val_df, user_vocab, item_vocab, histories, max_hist)
    print("  test:")
    test_t = make_tensors(test_df, user_vocab, item_vocab, histories, max_hist)
    print(f"  done ({time.time()-t0:.0f}s)")
    test_cold_mask = test_df["is_cold_start_item"].values.astype(bool)

    train_loader = DataLoader(TensorDataset(*train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_t), batch_size=BATCH_SIZE * 4, shuffle=False)
    test_loader = DataLoader(TensorDataset(*test_t), batch_size=BATCH_SIZE * 4, shuffle=False)

    print(f"\nDevice: {DEVICE}")
    model = DynamicGraphIDOnly(n_users, n_items, USER_EMBED_DIM, ITEM_EMBED_DIM, HIDDEN).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    bce = nn.BCELoss()

    best_val_auc = -1.0
    best_epoch = -1
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, bce)
        preds, labels = predict(model, val_loader)
        val_metrics = compute_metrics(preds, labels)
        print(f"Epoch {epoch:>2} | train_loss={train_loss:.5f} | "
              f"val AUC={val_metrics.get('auc', float('nan')):.4f} | "
              f"decay={F.softplus(model.temporal_attn.raw_decay).item():.4f} | "
              f"{time.time()-t0:.0f}s")

        val_auc = val_metrics.get("auc", -1.0)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_left = PATIENCE
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping. Best epoch: {best_epoch}")
                break

    print(f"\nLoading best checkpoint (epoch {best_epoch}) for final evaluation ...")
    model.load_state_dict(torch.load(checkpoint_path))

    results = {"seed": seed, "best_epoch": best_epoch, "max_history": max_hist,
               "n_users_train_vocab": n_users, "n_items_train_vocab": n_items,
               "learned_decay_rate": F.softplus(model.temporal_attn.raw_decay).item()}

    preds, labels = predict(model, val_loader)
    results["val"] = compute_metrics(preds, labels)

    preds, labels = predict(model, test_loader)
    results["test_overall"] = compute_metrics(preds, labels)
    results["test_seen_items"] = compute_metrics(preds, labels, mask=~test_cold_mask)
    results["test_cold_start_items"] = compute_metrics(preds, labels, mask=test_cold_mask)

    print("\n" + "=" * 70)
    print("FINAL RESULTS (Amazon Video_Games — V4-ID-Only, pure-ID dynamic graph ablation)")
    print("=" * 70)
    print(json.dumps(results, indent=2))

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    print(f"\nCompare against baseline_results/amazon_id_baseline_metrics.json (same ID "
          f"representation, no graph) to isolate the graph aggregator's own contribution.")


if __name__ == "__main__":
    main()

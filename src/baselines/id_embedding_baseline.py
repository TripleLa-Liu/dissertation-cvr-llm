"""
Ali-CCP Baseline #1 — Minimal ID-Embedding Model (ESMM-style, run locally)
============================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

WHY THIS DESIGN
----------------
The k-core-filtered dataset currently only carries scalar item_id/user_id
(the full categorical feature blob hasn't been extracted yet — see "Known
gap" in README). Rather than wait, this trains the simplest possible
baseline on exactly what's available: user_id + item_id embeddings. Two
things this doubles as, beyond "first working number":
  1. The pure-ID reference point for RQ1 — once the LLM Encoder (V1)
     replaces/augments these ID embeddings with text-derived ones, this is
     the number it needs to beat, on the SAME train/val/test split.
  2. A concrete measurement of the cold-start blind spot: 42.86% of the
     official test set involves items never seen in training
     (is_cold_start_item=1, see README). This baseline can only fall back
     to an UNK item embedding for those rows — we report metrics split by
     seen vs. cold-start specifically to quantify how much that costs.

ARCHITECTURE — ESMM-style entire-space multi-task model (Ma et al., SIGIR
2018, already in our literature review). Chosen over a single CTR-only
model because CVR prediction is the dissertation's actual target quantity,
and naively training a CVR model only on clicked rows suffers the sample
selection bias ESMM was specifically designed to avoid:
  - Shared embeddings: Embedding(user_id), Embedding(item_id), concatenated.
  - Two MLP towers on top of the shared embedding: CTR tower -> p(click),
    CVR tower -> p(purchase | click).
  - p(purchase | impression) = p_ctr * p_cvr  ("CTCVR"), estimated over the
    ENTIRE space (all impressions), not just clicked ones — this is exactly
    ESMM's fix for sample selection bias.
  - Loss = BCE(p_ctr, click_label) + BCE(p_ctr * p_cvr, purchase_label)

UNKNOWN IDs: vocab is built from TRAIN ONLY. Any user_id/item_id in
val/test not seen in train (including all cold-start test items) maps to
index 0 (a shared UNK embedding) — this is the realistic deployment
scenario and is exactly the mechanism whose cost we're measuring.

USAGE
-----
1. Make sure aliccp_train_split.csv, aliccp_val_split.csv (from
   split_train_val.py) and aliccp_test_filtered_joined.csv (from
   filter_test_and_join.py) are all in WORK_DIR below.
2. pip install torch scikit-learn pandas   (if not already installed)
3. Run: python id_embedding_baseline.py
   Expected runtime on a GPU: a few minutes for ~10 epochs over 2.9M rows.
   On CPU: expect 10-30x slower — reduce EPOCHS or BATCH_SIZE if needed.
4. Send me the printed final metrics (val + test overall + test seen/
   cold-start breakdown) — I'll write these into README/Overleaf and we'll
   use the saved vocab + checkpoint as the comparison point for V1.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    raise SystemExit("Please install scikit-learn: pip install scikit-learn")

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
WORK_DIR = r"E:\BaiduNetdiskDownload\Dataset\_processed"
RESULTS_DIR = r"E:\BaiduNetdiskDownload\Dataset\_processed\baseline_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(WORK_DIR, "aliccp_train_split.csv")
VAL_PATH = os.path.join(WORK_DIR, "aliccp_val_split.csv")
TEST_PATH = os.path.join(WORK_DIR, "aliccp_test_filtered_joined.csv")

VOCAB_PATH = os.path.join(RESULTS_DIR, "id_vocab.pkl")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "esmm_id_baseline.pt")
METRICS_PATH = os.path.join(RESULTS_DIR, "esmm_id_baseline_metrics.json")

EMBED_DIM = 32
HIDDEN = [128, 64]
BATCH_SIZE = 4096
EPOCHS = 15
PATIENCE = 3           # early stopping patience, measured on val CTCVR AUC
LR = 1e-3
WEIGHT_DECAY = 1e-6
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# Data loading + vocab
# ------------------------------------------------------------------
def load_csv(path):
    df = pd.read_csv(path, dtype={"user_id": str, "item_id": str,
                                   "click": "int8", "purchase": "int8"})
    df["user_id"] = df["user_id"].fillna("__MISSING__")
    return df


def build_vocab(series):
    """index 0 reserved for UNK; real ids start at 1."""
    uniques = series.unique()
    vocab = {v: i + 1 for i, v in enumerate(uniques)}
    return vocab


def encode(series, vocab):
    return series.map(lambda x: vocab.get(x, 0)).astype("int64").values


def make_tensors(df, user_vocab, item_vocab):
    u = encode(df["user_id"], user_vocab)
    i = encode(df["item_id"], item_vocab)
    click = df["click"].values.astype("float32")
    purchase = df["purchase"].values.astype("float32")
    return (torch.from_numpy(u), torch.from_numpy(i),
            torch.from_numpy(click), torch.from_numpy(purchase))


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
class ESMM(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=32, hidden=(128, 64)):
        super().__init__()
        self.user_emb = nn.Embedding(n_users + 1, embed_dim, padding_idx=0)
        self.item_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.ctr_tower = self._make_tower(embed_dim * 2, hidden)
        self.cvr_tower = self._make_tower(embed_dim * 2, hidden)

    @staticmethod
    def _make_tower(in_dim, hidden):
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1)]
            d = h
        layers += [nn.Linear(d, 1)]
        return nn.Sequential(*layers)

    def forward(self, u, i):
        x = torch.cat([self.user_emb(u), self.item_emb(i)], dim=1)
        p_ctr = torch.sigmoid(self.ctr_tower(x)).squeeze(-1)
        p_cvr = torch.sigmoid(self.cvr_tower(x)).squeeze(-1)
        p_ctcvr = p_ctr * p_cvr
        return p_ctr, p_cvr, p_ctcvr


# ------------------------------------------------------------------
# Train / eval
# ------------------------------------------------------------------
def train_epoch(model, loader, optimizer, bce):
    model.train()
    total_loss = 0.0
    n = 0
    for u, i, click, purchase in loader:
        u, i, click, purchase = u.to(DEVICE), i.to(DEVICE), click.to(DEVICE), purchase.to(DEVICE)
        optimizer.zero_grad()
        p_ctr, _p_cvr, p_ctcvr = model(u, i)
        loss = bce(p_ctr, click) + bce(p_ctcvr, purchase)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(u)
        n += len(u)
    return total_loss / n


@torch.no_grad()
def predict(model, loader):
    model.eval()
    p_ctrs, p_cvrs, p_ctcvrs, clicks, purchases = [], [], [], [], []
    for u, i, click, purchase in loader:
        u, i = u.to(DEVICE), i.to(DEVICE)
        p_ctr, p_cvr, p_ctcvr = model(u, i)
        p_ctrs.append(p_ctr.cpu().numpy())
        p_cvrs.append(p_cvr.cpu().numpy())
        p_ctcvrs.append(p_ctcvr.cpu().numpy())
        clicks.append(click.numpy())
        purchases.append(purchase.numpy())
    return (np.concatenate(p_ctrs), np.concatenate(p_cvrs), np.concatenate(p_ctcvrs),
            np.concatenate(clicks), np.concatenate(purchases))


def compute_metrics(p_ctr, p_cvr, p_ctcvr, click, purchase, mask=None):
    if mask is not None:
        p_ctr, p_cvr, p_ctcvr = p_ctr[mask], p_cvr[mask], p_ctcvr[mask]
        click, purchase = click[mask], purchase[mask]
    out = {"n_rows": int(len(click))}
    if len(np.unique(click)) > 1:
        out["ctr_auc"] = float(roc_auc_score(click, p_ctr))
    clicked = click == 1
    if clicked.sum() > 1 and len(np.unique(purchase[clicked])) > 1:
        out["cvr_auc_post_click"] = float(roc_auc_score(purchase[clicked], p_cvr[clicked]))
    if len(np.unique(purchase)) > 1:
        out["ctcvr_auc"] = float(roc_auc_score(purchase, p_ctcvr))
    out["ctr"] = float(click.mean())
    out["cvr_post_click"] = float(purchase[clicked].mean()) if clicked.sum() else None
    return out


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading train/val/test CSVs ...")
    t0 = time.time()
    train_df = load_csv(TRAIN_PATH)
    val_df = load_csv(VAL_PATH)
    test_df = load_csv(TEST_PATH)
    print(f"  train={len(train_df):,} val={len(val_df):,} test={len(test_df):,} "
          f"({time.time()-t0:.0f}s)")

    print("Building vocab from TRAIN only ...")
    user_vocab = build_vocab(train_df["user_id"])
    item_vocab = build_vocab(train_df["item_id"])
    n_users, n_items = len(user_vocab), len(item_vocab)
    print(f"  n_users={n_users:,} n_items={n_items:,}")

    import pickle
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump({"user_vocab": user_vocab, "item_vocab": item_vocab}, f)
    print(f"  Saved vocab to {VOCAB_PATH} (reused for V1 so ID embeddings stay comparable)")

    train_t = make_tensors(train_df, user_vocab, item_vocab)
    val_t = make_tensors(val_df, user_vocab, item_vocab)
    test_t = make_tensors(test_df, user_vocab, item_vocab)
    test_cold_mask = test_df["is_cold_start_item"].values.astype(bool)

    train_loader = DataLoader(TensorDataset(*train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_t), batch_size=BATCH_SIZE * 4, shuffle=False)
    test_loader = DataLoader(TensorDataset(*test_t), batch_size=BATCH_SIZE * 4, shuffle=False)

    print(f"\nDevice: {DEVICE}")
    model = ESMM(n_users, n_items, EMBED_DIM, HIDDEN).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    bce = nn.BCELoss()

    best_val_ctcvr_auc = -1.0
    best_epoch = -1
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, bce)
        p_ctr, p_cvr, p_ctcvr, click, purchase = predict(model, val_loader)
        val_metrics = compute_metrics(p_ctr, p_cvr, p_ctcvr, click, purchase)
        print(f"Epoch {epoch:>2} | train_loss={train_loss:.5f} | "
              f"val CTR-AUC={val_metrics.get('ctr_auc', float('nan')):.4f} | "
              f"val CVR-AUC(post-click)={val_metrics.get('cvr_auc_post_click', float('nan')):.4f} | "
              f"val CTCVR-AUC={val_metrics.get('ctcvr_auc', float('nan')):.4f} | "
              f"{time.time()-t0:.0f}s")

        val_ctcvr_auc = val_metrics.get("ctcvr_auc", -1.0)
        if val_ctcvr_auc > best_val_ctcvr_auc:
            best_val_ctcvr_auc = val_ctcvr_auc
            best_epoch = epoch
            patience_left = PATIENCE
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping (no val CTCVR-AUC improvement for {PATIENCE} epochs). "
                      f"Best epoch: {best_epoch}")
                break

    print(f"\nLoading best checkpoint (epoch {best_epoch}) for final evaluation ...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH))

    results = {"best_epoch": best_epoch, "embed_dim": EMBED_DIM, "hidden": HIDDEN,
               "n_users_train_vocab": n_users, "n_items_train_vocab": n_items}

    p_ctr, p_cvr, p_ctcvr, click, purchase = predict(model, val_loader)
    results["val"] = compute_metrics(p_ctr, p_cvr, p_ctcvr, click, purchase)

    p_ctr, p_cvr, p_ctcvr, click, purchase = predict(model, test_loader)
    results["test_overall"] = compute_metrics(p_ctr, p_cvr, p_ctcvr, click, purchase)
    results["test_seen_items"] = compute_metrics(p_ctr, p_cvr, p_ctcvr, click, purchase,
                                                   mask=~test_cold_mask)
    results["test_cold_start_items"] = compute_metrics(p_ctr, p_cvr, p_ctcvr, click, purchase,
                                                          mask=test_cold_mask)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(json.dumps(results, indent=2))

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {METRICS_PATH}")
    print(f"Saved best checkpoint to {CHECKPOINT_PATH}")
    print("\nSend me the printed FINAL RESULTS block above — I'll write it into "
          "README/Overleaf and we'll use this as the comparison point for LLM Encoder V1.")


if __name__ == "__main__":
    main()

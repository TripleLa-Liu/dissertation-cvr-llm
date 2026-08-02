"""
Amazon Reviews'23 (Video_Games) Baseline — ID-Embedding Binary Classifier
============================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

Pure-ID reference point for the "Text/no text dataset" experiment, on the
same train/val/test split used by amazon_text_embedding.py, so results are
directly comparable. Unlike the Ali-CCP baseline (id_embedding_baseline.py),
this is a single binary classification task, not ESMM's two-stage CTR/CVR
structure — amazon_build_dataset.py's negative sampling already collapses
the problem to "did this user interact with this item" (label 1/0), since
Amazon Reviews'23 has no natural impression/no-click signal to build a
CTR stage from.

Architecture: user_id + item_id learned embeddings, concatenated, through
an MLP, BCE loss against label. Vocab built from train only; unseen
user_id/item_id in val/test map to a shared UNK embedding (index 0) — same
convention as the Ali-CCP baseline, and the same mechanism
is_cold_start_item in the data measures the cost of.

Requires amazon_train.csv / amazon_val.csv / amazon_test.csv (from
amazon_build_dataset.py) in WORK_DIR/amazon/processed/.
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
WORK_DIR = r"D:\Study\migration_package\processed_data"
PROCESSED_DIR = os.path.join(WORK_DIR, "amazon", "processed")
RESULTS_DIR = os.path.join(WORK_DIR, "amazon", "baseline_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(PROCESSED_DIR, "amazon_train.csv")
VAL_PATH = os.path.join(PROCESSED_DIR, "amazon_val.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "amazon_test.csv")

VOCAB_PATH = os.path.join(RESULTS_DIR, "id_vocab.pkl")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "amazon_id_baseline.pt")
METRICS_PATH = os.path.join(RESULTS_DIR, "amazon_id_baseline_metrics.json")

EMBED_DIM = 32
HIDDEN = [128, 64]
BATCH_SIZE = 1024
EPOCHS = 20
PATIENCE = 3
LR = 1e-3
WEIGHT_DECAY = 1e-6
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
def load_csv(path):
    return pd.read_csv(path, dtype={"user_id": str, "item_id": str,
                                     "label": "int8", "is_cold_start_item": "int8"})


def build_vocab(series):
    uniques = series.unique()
    return {v: i + 1 for i, v in enumerate(uniques)}  # index 0 reserved for UNK


def encode(series, vocab):
    return series.map(lambda x: vocab.get(x, 0)).astype("int64").values


def make_tensors(df, user_vocab, item_vocab):
    u = encode(df["user_id"], user_vocab)
    i = encode(df["item_id"], item_vocab)
    label = df["label"].values.astype("float32")
    return torch.from_numpy(u), torch.from_numpy(i), torch.from_numpy(label)


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
class IDBaseline(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=32, hidden=(128, 64)):
        super().__init__()
        self.user_emb = nn.Embedding(n_users + 1, embed_dim, padding_idx=0)
        self.item_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        layers = []
        d = embed_dim * 2
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.tower = nn.Sequential(*layers)

    def forward(self, u, i):
        x = torch.cat([self.user_emb(u), self.item_emb(i)], dim=1)
        return torch.sigmoid(self.tower(x)).squeeze(-1)


# ------------------------------------------------------------------
# Train / eval
# ------------------------------------------------------------------
def train_epoch(model, loader, optimizer, bce):
    model.train()
    total_loss, n = 0.0, 0
    for u, i, label in loader:
        u, i, label = u.to(DEVICE), i.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        p = model(u, i)
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
    for u, i, label in loader:
        u, i = u.to(DEVICE), i.to(DEVICE)
        preds.append(model(u, i).cpu().numpy())
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

    train_t = make_tensors(train_df, user_vocab, item_vocab)
    val_t = make_tensors(val_df, user_vocab, item_vocab)
    test_t = make_tensors(test_df, user_vocab, item_vocab)
    test_cold_mask = test_df["is_cold_start_item"].values.astype(bool)

    train_loader = DataLoader(TensorDataset(*train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_t), batch_size=BATCH_SIZE * 4, shuffle=False)
    test_loader = DataLoader(TensorDataset(*test_t), batch_size=BATCH_SIZE * 4, shuffle=False)

    print(f"\nDevice: {DEVICE}")
    model = IDBaseline(n_users, n_items, EMBED_DIM, HIDDEN).to(DEVICE)
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
              f"{time.time()-t0:.0f}s")

        val_auc = val_metrics.get("auc", -1.0)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_left = PATIENCE
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping. Best epoch: {best_epoch}")
                break

    print(f"\nLoading best checkpoint (epoch {best_epoch}) for final evaluation ...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH))

    results = {"best_epoch": best_epoch, "embed_dim": EMBED_DIM, "hidden": HIDDEN,
               "n_users_train_vocab": n_users, "n_items_train_vocab": n_items}

    preds, labels = predict(model, val_loader)
    results["val"] = compute_metrics(preds, labels)

    preds, labels = predict(model, test_loader)
    results["test_overall"] = compute_metrics(preds, labels)
    results["test_seen_items"] = compute_metrics(preds, labels, mask=~test_cold_mask)
    results["test_cold_start_items"] = compute_metrics(preds, labels, mask=test_cold_mask)

    print("\n" + "=" * 70)
    print("FINAL RESULTS (Amazon Video_Games — ID-embedding baseline)")
    print("=" * 70)
    print(json.dumps(results, indent=2))

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {METRICS_PATH}")
    print(f"Saved best checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()

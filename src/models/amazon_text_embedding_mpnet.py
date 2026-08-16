"""
Amazon Reviews'23 (Video_Games) V1-MPNet — item-only text replace, MPNet encoder
============================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

"Different Embedders" experiment, extended to Amazon (2026-08-15 supervisor
request — the MiniLM-vs-MPNet comparison had only ever been run on Ali-CCP,
leaving Amazon's model-variant x embedder x dataset matrix incomplete).
Identical to amazon_text_embedding.py, with the frozen sentence-transformer
swapped from all-MiniLM-L6-v2 (384-dim) to all-mpnet-base-v2 (768-dim).

Everything else (architecture, train/val/test split, evaluation protocol)
is unchanged from amazon_text_embedding.py — see that file for full design
rationale. Writes to a separate text_embedding_mpnet_results/ directory so
this run doesn't overwrite the MiniLM-based V1 checkpoint.

Requires amazon_item_text.csv (from amazon_build_dataset.py) and
sentence-transformers installed.
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
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    raise SystemExit("Please install scikit-learn: pip install scikit-learn")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit("Please install sentence-transformers: pip install sentence-transformers")

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
# Local Mac run (2026-08-15, Amazon MPNet gap) — see amazon_download.py.
WORK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "migration_package", "processed_data")
PROCESSED_DIR = os.path.join(WORK_DIR, "amazon", "processed")
RESULTS_DIR = os.path.join(WORK_DIR, "amazon", "text_embedding_mpnet_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(PROCESSED_DIR, "amazon_train.csv")
VAL_PATH = os.path.join(PROCESSED_DIR, "amazon_val.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "amazon_test.csv")
ITEM_TEXT_PATH = os.path.join(PROCESSED_DIR, "amazon_item_text.csv")

# independent cache — separate from text_embedding_results/ (MiniLM) so both
# can be compared side by side without either overwriting the other
ITEM_EMBED_CACHE = os.path.join(RESULTS_DIR, "item_llm_embeddings.pkl")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "amazon_text_embedding_mpnet.pt")
METRICS_PATH = os.path.join(RESULTS_DIR, "amazon_text_embedding_mpnet_metrics.json")

LM_NAME = "all-mpnet-base-v2"   # only change vs. amazon_text_embedding.py (was all-MiniLM-L6-v2)
USER_EMBED_DIM = 32
ADAPTER_DIM = 32
HIDDEN = [128, 64]
BATCH_SIZE = 1024
EPOCHS = 20
PATIENCE = 3
LR = 1e-3
WEIGHT_DECAY = 1e-6
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """--seed lets this script be rerun under multiple random seeds for
    mean/std reporting, matching every other multi-seed model in this
    project. Default (42) keeps the original unsuffixed checkpoint/metrics
    filenames."""
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED,
                   help="random seed (default: 42)")
    return p.parse_args()


def seed_suffixed(path, seed):
    if seed == SEED:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_seed{seed}{ext}"


# ------------------------------------------------------------------
# Frozen item LLM embeddings
# ------------------------------------------------------------------
def build_item_llm_embeddings():
    print(f"Loading {ITEM_TEXT_PATH} ...")
    df = pd.read_csv(ITEM_TEXT_PATH, dtype={"item_id": str, "real_text": str})
    print(f"  {len(df):,} distinct items")

    if os.path.exists(ITEM_EMBED_CACHE):
        with open(ITEM_EMBED_CACHE, "rb") as f:
            cached = pickle.load(f)
        # See amazon_text_embedding.py's identical guard: the cache isn't
        # namespaced by CATEGORY/dataset, so a stale cache would silently
        # produce a constant item representation for everything.
        cached_ids = set(cached["item_id_to_row"].keys())
        current_ids = set(df["item_id"])
        if cached_ids >= current_ids:
            print(f"Loading cached item LLM embeddings from {ITEM_EMBED_CACHE} ...")
            return cached
        else:
            print(f"Cached embeddings at {ITEM_EMBED_CACHE} don't cover the current "
                  f"item set ({len(cached_ids & current_ids):,}/{len(current_ids):,} "
                  f"items match) — stale cache from a different CATEGORY/run, "
                  f"re-encoding instead of trusting it.")

    print(f"Loading sentence-transformer '{LM_NAME}' (frozen) ...")
    model = SentenceTransformer(LM_NAME, device=str(DEVICE))
    print("Encoding item text ...")
    t0 = time.time()
    embeddings = model.encode(df["real_text"].fillna("").tolist(), batch_size=256,
                               show_progress_bar=True, convert_to_numpy=True).astype("float32")
    print(f"  Encoded {len(df):,} in {time.time()-t0:.0f}s, dim={embeddings.shape[1]}")

    item_id_to_row = {v: i for i, v in enumerate(df["item_id"])}
    embeddings = np.vstack([embeddings, np.zeros((1, embeddings.shape[1]), dtype="float32")])
    unk_row = embeddings.shape[0] - 1
    result = {"item_id_to_row": item_id_to_row, "embeddings": embeddings, "unk_row": unk_row}
    with open(ITEM_EMBED_CACHE, "wb") as f:
        pickle.dump(result, f)
    print(f"Cached to {ITEM_EMBED_CACHE}")
    return result


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


def make_tensors(df, user_vocab, item_id_to_row, unk_row):
    u = encode(df["user_id"], user_vocab)
    i_row = df["item_id"].map(lambda x: item_id_to_row.get(x, unk_row)).astype("int64").values.copy()
    label = df["label"].values.astype("float32")
    return torch.from_numpy(u), torch.from_numpy(i_row), torch.from_numpy(label)


# ------------------------------------------------------------------
# Model (identical to TextEmbeddingModel in amazon_text_embedding.py —
# lm_dim is read dynamically from the embedding matrix, so this class
# needs no changes for the larger 768-dim MPNet vectors)
# ------------------------------------------------------------------
class TextEmbeddingModel(nn.Module):
    def __init__(self, n_users, item_embed_matrix, user_dim=32, adapter_dim=32, hidden=(128, 64)):
        super().__init__()
        self.user_emb = nn.Embedding(n_users + 1, user_dim, padding_idx=0)
        n_items, lm_dim = item_embed_matrix.shape
        self.item_emb_frozen = nn.Embedding.from_pretrained(
            torch.from_numpy(item_embed_matrix), freeze=True)
        self.item_adapter = nn.Sequential(
            nn.Linear(lm_dim, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, adapter_dim))
        layers = []
        d = user_dim + adapter_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.tower = nn.Sequential(*layers)

    def forward(self, u, i_row):
        item_repr = self.item_adapter(self.item_emb_frozen(i_row))
        x = torch.cat([self.user_emb(u), item_repr], dim=1)
        return torch.sigmoid(self.tower(x)).squeeze(-1)


# ------------------------------------------------------------------
# Train / eval (identical logic to amazon_text_embedding.py)
# ------------------------------------------------------------------
def train_epoch(model, loader, optimizer, bce):
    model.train()
    total_loss, n = 0.0, 0
    for u, i_row, label in loader:
        u, i_row, label = u.to(DEVICE), i_row.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        p = model(u, i_row)
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
    for u, i_row, label in loader:
        u, i_row = u.to(DEVICE), i_row.to(DEVICE)
        preds.append(model(u, i_row).cpu().numpy())
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

    item_llm_data = build_item_llm_embeddings()
    item_id_to_row = item_llm_data["item_id_to_row"]
    item_embed_matrix = item_llm_data["embeddings"]
    unk_row = item_llm_data["unk_row"]

    print("\nLoading train/val/test CSVs ...")
    t0 = time.time()
    train_df = load_csv(TRAIN_PATH)
    val_df = load_csv(VAL_PATH)
    test_df = load_csv(TEST_PATH)
    print(f"  train={len(train_df):,} val={len(val_df):,} test={len(test_df):,} "
          f"({time.time()-t0:.0f}s)")

    print("Building user vocab from TRAIN only ...")
    user_vocab = build_vocab(train_df["user_id"])
    n_users = len(user_vocab)
    print(f"  n_users={n_users:,}")

    train_t = make_tensors(train_df, user_vocab, item_id_to_row, unk_row)
    val_t = make_tensors(val_df, user_vocab, item_id_to_row, unk_row)
    test_t = make_tensors(test_df, user_vocab, item_id_to_row, unk_row)
    test_cold_mask = test_df["is_cold_start_item"].values.astype(bool)

    train_loader = DataLoader(TensorDataset(*train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_t), batch_size=BATCH_SIZE * 4, shuffle=False)
    test_loader = DataLoader(TensorDataset(*test_t), batch_size=BATCH_SIZE * 4, shuffle=False)

    print(f"\nDevice: {DEVICE}")
    model = TextEmbeddingModel(n_users, item_embed_matrix, USER_EMBED_DIM, ADAPTER_DIM,
                                HIDDEN).to(DEVICE)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
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
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping. Best epoch: {best_epoch}")
                break

    print(f"\nLoading best checkpoint (epoch {best_epoch}) for final evaluation ...")
    model.load_state_dict(torch.load(checkpoint_path))

    results = {"seed": seed, "best_epoch": best_epoch, "lm_name": LM_NAME, "n_users_train_vocab": n_users}

    preds, labels = predict(model, val_loader)
    results["val"] = compute_metrics(preds, labels)

    preds, labels = predict(model, test_loader)
    results["test_overall"] = compute_metrics(preds, labels)
    results["test_seen_items"] = compute_metrics(preds, labels, mask=~test_cold_mask)
    results["test_cold_start_items"] = compute_metrics(preds, labels, mask=test_cold_mask)

    print("\n" + "=" * 70)
    print("FINAL RESULTS (Amazon Video_Games — V1-MPNet real-text item embedding)")
    print("=" * 70)
    print(json.dumps(results, indent=2))

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    print(f"\nCompare against text_embedding_results/amazon_text_embedding_metrics.json (MiniLM) "
          f"to see whether encoder quality moved the numbers.")


if __name__ == "__main__":
    main()

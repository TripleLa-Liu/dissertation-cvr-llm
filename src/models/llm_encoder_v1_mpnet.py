"""
Ali-CCP LLM Encoder V1-MPNet — item-only text replace, MPNet encoder
====================================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

"Different Embedders" experiment, extended to V1 (2026-08-04 supervisor
request for a more robust comparison matrix across model variant x
embedder x dataset). Identical to llm_encoder_v1.py, with the frozen
sentence-transformer swapped from all-MiniLM-L6-v2 (384-dim) to
all-mpnet-base-v2 (768-dim). Previously the MPNet swap was only tested on
V2 (llm_encoder_v2_mpnet.py), since V2 was the only variant competitive
with the ID baseline; this fills in the same comparison for V1 (the
REPLACE pattern) so the "does a bigger encoder help" question can be
answered across both architectural patterns, not just the ALIGN one.

Everything else (architecture, train/val/test split, evaluation protocol)
is unchanged from llm_encoder_v1.py — see that file for full design
rationale. Writes to a separate v1_mpnet_results/ directory so this run
doesn't overwrite the MiniLM-based V1 checkpoint.

Requires item_pseudo_text.csv (from extract_item_pseudo_text.py).
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
WORK_DIR = r"D:\Study\migration_package\processed_data"
RESULTS_DIR = r"D:\Study\migration_package\processed_data\v1_mpnet_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(WORK_DIR, "aliccp_train_split.csv")
VAL_PATH = os.path.join(WORK_DIR, "aliccp_val_split.csv")
TEST_PATH = os.path.join(WORK_DIR, "aliccp_test_filtered_joined.csv")
ITEM_TEXT_PATH = os.path.join(WORK_DIR, "item_pseudo_text.csv")

# independent cache — separate from v1_results/ (MiniLM) so both can be
# compared side by side without either overwriting the other
ITEM_EMBED_CACHE = os.path.join(RESULTS_DIR, "item_llm_embeddings.pkl")
VOCAB_PATH = os.path.join(RESULTS_DIR, "id_vocab_v1_mpnet.pkl")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "esmm_llm_v1_mpnet.pt")
METRICS_PATH = os.path.join(RESULTS_DIR, "esmm_llm_v1_mpnet_metrics.json")

LM_NAME = "all-mpnet-base-v2"   # only change vs. llm_encoder_v1.py (was all-MiniLM-L6-v2)
USER_EMBED_DIM = 32
ADAPTER_DIM = 32
HIDDEN = [128, 64]
BATCH_SIZE = 4096
EPOCHS = 15
PATIENCE = 3
LR = 1e-3
WEIGHT_DECAY = 1e-6
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """--seed lets this script be rerun under multiple random seeds for
    mean/std reporting (2026-08-04 supervisor request). Default (42) keeps
    the original unsuffixed checkpoint/metrics filenames."""
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED,
                   help="random seed (default: 42, the original single-run seed)")
    return p.parse_args()


def seed_suffixed(path, seed):
    if seed == SEED:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_seed{seed}{ext}"


# ------------------------------------------------------------------
# Item LLM embeddings (frozen, precomputed once)
# ------------------------------------------------------------------
def build_item_embeddings():
    if os.path.exists(ITEM_EMBED_CACHE):
        print(f"Loading cached item LLM embeddings from {ITEM_EMBED_CACHE} ...")
        with open(ITEM_EMBED_CACHE, "rb") as f:
            return pickle.load(f)

    print(f"Loading {ITEM_TEXT_PATH} ...")
    item_text_df = pd.read_csv(ITEM_TEXT_PATH, dtype={"item_id": str, "pseudo_text": str})
    print(f"  {len(item_text_df):,} distinct items")

    print(f"Loading sentence-transformer '{LM_NAME}' (frozen, no fine-tuning) ...")
    model = SentenceTransformer(LM_NAME, device=str(DEVICE))

    print("Encoding pseudo-text for all items (batched) ...")
    t0 = time.time()
    texts = item_text_df["pseudo_text"].tolist()
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True,
                               convert_to_numpy=True)
    print(f"  Encoded {len(texts):,} items in {time.time()-t0:.0f}s, "
          f"embedding dim={embeddings.shape[1]}")

    item_id_to_row = {iid: i for i, iid in enumerate(item_text_df["item_id"])}
    embeddings = np.vstack([embeddings.astype("float32"),
                             np.zeros((1, embeddings.shape[1]), dtype="float32")])
    unk_row = embeddings.shape[0] - 1
    result = {"item_id_to_row": item_id_to_row, "embeddings": embeddings, "unk_row": unk_row}
    with open(ITEM_EMBED_CACHE, "wb") as f:
        pickle.dump(result, f)
    print(f"Cached to {ITEM_EMBED_CACHE}")
    return result


# ------------------------------------------------------------------
# Data loading (user side identical to baseline)
# ------------------------------------------------------------------
def load_csv(path):
    df = pd.read_csv(path, dtype={"user_id": str, "item_id": str,
                                   "click": "int8", "purchase": "int8"})
    df["user_id"] = df["user_id"].fillna("__MISSING__")
    return df


def build_vocab(series):
    uniques = series.unique()
    return {v: i + 1 for i, v in enumerate(uniques)}


def encode(series, vocab):
    return series.map(lambda x: vocab.get(x, 0)).astype("int64").values.copy()


def encode_item_rows(series, item_id_to_row, unk_row):
    rows = series.map(lambda x: item_id_to_row.get(x, unk_row)).astype("int64").values.copy()
    n_missing = int((rows == unk_row).sum())
    if n_missing:
        print(f"  WARNING: {n_missing} rows have an item_id missing from "
              f"item_pseudo_text.csv — falling back to the zero-vector UNK row for these.")
    return rows


def make_tensors(df, user_vocab, item_id_to_row, unk_row):
    u = encode(df["user_id"], user_vocab)
    i_row = encode_item_rows(df["item_id"], item_id_to_row, unk_row)
    click = df["click"].values.astype("float32")
    purchase = df["purchase"].values.astype("float32")
    return (torch.from_numpy(u), torch.from_numpy(i_row),
            torch.from_numpy(click), torch.from_numpy(purchase))


# ------------------------------------------------------------------
# Model (identical to ESMM_LLM_V1 in llm_encoder_v1.py — lm_dim is read
# dynamically from the embedding matrix, so this class needs no changes
# for the larger 768-dim MPNet vectors)
# ------------------------------------------------------------------
class ESMM_LLM_V1(nn.Module):
    def __init__(self, n_users, item_embed_matrix, user_dim=32, adapter_dim=32,
                 hidden=(128, 64)):
        super().__init__()
        self.user_emb = nn.Embedding(n_users + 1, user_dim, padding_idx=0)

        n_items, lm_dim = item_embed_matrix.shape
        self.item_emb_frozen = nn.Embedding.from_pretrained(
            torch.from_numpy(item_embed_matrix), freeze=True)
        self.item_adapter = nn.Sequential(
            nn.Linear(lm_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, adapter_dim),
        )

        tower_in = user_dim + adapter_dim
        self.ctr_tower = self._make_tower(tower_in, hidden)
        self.cvr_tower = self._make_tower(tower_in, hidden)

    @staticmethod
    def _make_tower(in_dim, hidden):
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1)]
            d = h
        layers += [nn.Linear(d, 1)]
        return nn.Sequential(*layers)

    def forward(self, u, i_row):
        item_repr = self.item_adapter(self.item_emb_frozen(i_row))
        x = torch.cat([self.user_emb(u), item_repr], dim=1)
        p_ctr = torch.sigmoid(self.ctr_tower(x)).squeeze(-1)
        p_cvr = torch.sigmoid(self.cvr_tower(x)).squeeze(-1)
        p_ctcvr = p_ctr * p_cvr
        return p_ctr, p_cvr, p_ctcvr


# ------------------------------------------------------------------
# Train / eval (identical logic to llm_encoder_v1.py)
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
    args = parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    checkpoint_path = seed_suffixed(CHECKPOINT_PATH, seed)
    metrics_path = seed_suffixed(METRICS_PATH, seed)
    if seed != SEED:
        print(f"Running with --seed={seed} (non-default): outputs -> "
              f"{os.path.basename(checkpoint_path)}, {os.path.basename(metrics_path)}")

    item_data = build_item_embeddings()
    item_id_to_row = item_data["item_id_to_row"]
    item_embed_matrix = item_data["embeddings"]
    unk_row = item_data["unk_row"]

    print("\nLoading train/val/test CSVs ...")
    t0 = time.time()
    train_df = load_csv(TRAIN_PATH)
    val_df = load_csv(VAL_PATH)
    test_df = load_csv(TEST_PATH)
    print(f"  train={len(train_df):,} val={len(val_df):,} test={len(test_df):,} "
          f"({time.time()-t0:.0f}s)")

    print("Building user vocab from TRAIN only (same as baseline) ...")
    user_vocab = build_vocab(train_df["user_id"])
    n_users = len(user_vocab)
    print(f"  n_users={n_users:,}")
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump({"user_vocab": user_vocab}, f)

    train_t = make_tensors(train_df, user_vocab, item_id_to_row, unk_row)
    val_t = make_tensors(val_df, user_vocab, item_id_to_row, unk_row)
    test_t = make_tensors(test_df, user_vocab, item_id_to_row, unk_row)
    test_cold_mask = test_df["is_cold_start_item"].values.astype(bool)

    train_loader = DataLoader(TensorDataset(*train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_t), batch_size=BATCH_SIZE * 4, shuffle=False)
    test_loader = DataLoader(TensorDataset(*test_t), batch_size=BATCH_SIZE * 4, shuffle=False)

    print(f"\nDevice: {DEVICE}")
    model = ESMM_LLM_V1(n_users, item_embed_matrix, USER_EMBED_DIM, ADAPTER_DIM, HIDDEN).to(DEVICE)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
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
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping. Best epoch: {best_epoch}")
                break

    print(f"\nLoading best checkpoint (epoch {best_epoch}) for final evaluation ...")
    model.load_state_dict(torch.load(checkpoint_path))

    results = {"seed": seed, "best_epoch": best_epoch, "lm_name": LM_NAME, "adapter_dim": ADAPTER_DIM,
               "user_embed_dim": USER_EMBED_DIM, "hidden": HIDDEN, "n_users_train_vocab": n_users}

    p_ctr, p_cvr, p_ctcvr, click, purchase = predict(model, val_loader)
    results["val"] = compute_metrics(p_ctr, p_cvr, p_ctcvr, click, purchase)

    p_ctr, p_cvr, p_ctcvr, click, purchase = predict(model, test_loader)
    results["test_overall"] = compute_metrics(p_ctr, p_cvr, p_ctcvr, click, purchase)
    results["test_seen_items"] = compute_metrics(p_ctr, p_cvr, p_ctcvr, click, purchase,
                                                   mask=~test_cold_mask)
    results["test_cold_start_items"] = compute_metrics(p_ctr, p_cvr, p_ctcvr, click, purchase,
                                                          mask=test_cold_mask)

    print("\n" + "=" * 70)
    print("FINAL RESULTS (LLM Encoder V1-MPNet — frozen MPNet item text)")
    print("=" * 70)
    print(json.dumps(results, indent=2))

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    print(f"\nCompare against v1_results/esmm_llm_v1_metrics.json (MiniLM) "
          f"to see whether encoder quality moved the numbers.")


if __name__ == "__main__":
    main()

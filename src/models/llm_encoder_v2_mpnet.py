"""
Ali-CCP LLM Encoder V2-MPNet — RLMRec-style alignment, MPNet encoder
====================================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

"Different Embedders" experiment: identical to llm_encoder_v2_aligned.py
(V2), with the frozen sentence-transformer swapped from all-MiniLM-L6-v2
(384-dim) to all-mpnet-base-v2 (768-dim). V2 was chosen as the base for
this comparison because it was the only LLM-variant that matched or beat
the ID-embedding baseline overall, so it's the more informative test bed
for "does encoder quality matter" than V1/V1-Full, which already
underperform regardless of embedder.

MPNet is a stronger sentence encoder than MiniLM on most retrieval/STS
benchmarks (larger, slower, 2x the embedding dimensionality) but MiniLM
was chosen originally purely for speed. If results are materially better
here, MiniLM was a real bottleneck; if not, the anonymised-ID pseudo-text
ceiling (see README "LLM text feasibility" note) is the binding constraint
regardless of encoder quality — informative either way for RQ1.

Everything else (architecture, train/val/test split, alignment loss,
evaluation protocol) is unchanged from llm_encoder_v2_aligned.py — see
that file for full design rationale. Writes to a separate v2_mpnet_results/
directory so this run doesn't overwrite the MiniLM-based V2 checkpoint.

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
import torch.nn.functional as F
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
RESULTS_DIR = r"D:\Study\migration_package\processed_data\v2_mpnet_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(WORK_DIR, "aliccp_train_split.csv")
VAL_PATH = os.path.join(WORK_DIR, "aliccp_val_split.csv")
TEST_PATH = os.path.join(WORK_DIR, "aliccp_test_filtered_joined.csv")
ITEM_TEXT_PATH = os.path.join(WORK_DIR, "item_pseudo_text.csv")

# independent cache — separate from v2_results/ (MiniLM) so both can be
# compared side by side without either overwriting the other
ITEM_EMBED_CACHE = os.path.join(RESULTS_DIR, "item_llm_embeddings.pkl")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "esmm_v2_mpnet.pt")
METRICS_PATH = os.path.join(RESULTS_DIR, "esmm_v2_mpnet_metrics.json")

LM_NAME = "all-mpnet-base-v2"   # only change vs. llm_encoder_v2_aligned.py (was all-MiniLM-L6-v2)
USER_EMBED_DIM = 32
ITEM_EMBED_DIM = 32
HIDDEN = [128, 64]
BATCH_SIZE = 4096
EPOCHS = 15
PATIENCE = 3
LR = 1e-3
WEIGHT_DECAY = 1e-6
LAMBDA_ALIGN = 0.1          # weight of the contrastive alignment loss
CONTRASTIVE_TEMPERATURE = 0.1
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
# Frozen item LLM embeddings (same pattern as v1 / v2 scripts)
# ------------------------------------------------------------------
def build_item_llm_embeddings():
    if os.path.exists(ITEM_EMBED_CACHE):
        print(f"Loading cached item LLM embeddings from {ITEM_EMBED_CACHE} ...")
        with open(ITEM_EMBED_CACHE, "rb") as f:
            return pickle.load(f)

    print(f"Loading {ITEM_TEXT_PATH} ...")
    df = pd.read_csv(ITEM_TEXT_PATH, dtype={"item_id": str, "pseudo_text": str})
    print(f"  {len(df):,} distinct items")

    print(f"Loading sentence-transformer '{LM_NAME}' (frozen) ...")
    model = SentenceTransformer(LM_NAME, device=str(DEVICE))
    print("Encoding pseudo-text ...")
    t0 = time.time()
    embeddings = model.encode(df["pseudo_text"].tolist(), batch_size=256,
                               show_progress_bar=True, convert_to_numpy=True).astype("float32")
    print(f"  Encoded {len(df):,} in {time.time()-t0:.0f}s, dim={embeddings.shape[1]}")

    item_id_to_llm_row = {v: i for i, v in enumerate(df["item_id"])}
    embeddings = np.vstack([embeddings, np.zeros((1, embeddings.shape[1]), dtype="float32")])
    unk_row = embeddings.shape[0] - 1
    result = {"item_id_to_llm_row": item_id_to_llm_row, "embeddings": embeddings, "unk_row": unk_row}
    with open(ITEM_EMBED_CACHE, "wb") as f:
        pickle.dump(result, f)
    print(f"Cached to {ITEM_EMBED_CACHE}")
    return result


# ------------------------------------------------------------------
# Data loading
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


def encode_llm_rows(series, id_to_llm_row, unk_row):
    rows = series.map(lambda x: id_to_llm_row.get(x, unk_row)).astype("int64").values.copy()
    return rows


def make_tensors(df, user_vocab, item_vocab, item_id_to_llm_row, llm_unk_row):
    u = encode(df["user_id"], user_vocab)
    i_idx = encode(df["item_id"], item_vocab)                       # 0 = unseen in train
    i_llm_row = encode_llm_rows(df["item_id"], item_id_to_llm_row, llm_unk_row)  # always valid
    click = df["click"].values.astype("float32")
    purchase = df["purchase"].values.astype("float32")
    return (torch.from_numpy(u), torch.from_numpy(i_idx), torch.from_numpy(i_llm_row),
            torch.from_numpy(click), torch.from_numpy(purchase))


# ------------------------------------------------------------------
# Model (identical to ESMM_V2_Aligned in llm_encoder_v2_aligned.py —
# lm_dim is read dynamically from the embedding matrix, so this class
# needs no changes for the larger 768-dim MPNet vectors)
# ------------------------------------------------------------------
class ESMM_V2_Aligned(nn.Module):
    def __init__(self, n_users, n_items, item_llm_embed_matrix, user_dim=32, item_dim=32,
                 hidden=(128, 64)):
        super().__init__()
        self.user_emb = nn.Embedding(n_users + 1, user_dim, padding_idx=0)
        self.item_emb = nn.Embedding(n_items + 1, item_dim, padding_idx=0)

        self.item_llm_frozen = nn.Embedding.from_pretrained(
            torch.from_numpy(item_llm_embed_matrix), freeze=True)
        lm_dim = item_llm_embed_matrix.shape[1]
        self.item_llm_proj = nn.Sequential(
            nn.Linear(lm_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, item_dim),
        )

        tower_in = user_dim + item_dim
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

    def item_representation(self, item_idx, item_llm_row):
        """idx != 0 (seen in train) -> learned ID embedding.
        idx == 0 (cold-start, unseen in train) -> projected frozen LLM embedding.
        A masked blend covers both train (idx always != 0) and test (mixed)
        with one code path."""
        id_repr = self.item_emb(item_idx)
        llm_repr = self.item_llm_proj(self.item_llm_frozen(item_llm_row))
        use_id = (item_idx != 0).float().unsqueeze(-1)
        return use_id * id_repr + (1 - use_id) * llm_repr

    def forward(self, u, item_idx, item_llm_row):
        item_repr = self.item_representation(item_idx, item_llm_row)
        x = torch.cat([self.user_emb(u), item_repr], dim=1)
        p_ctr = torch.sigmoid(self.ctr_tower(x)).squeeze(-1)
        p_cvr = torch.sigmoid(self.cvr_tower(x)).squeeze(-1)
        p_ctcvr = p_ctr * p_cvr
        return p_ctr, p_cvr, p_ctcvr

    def alignment_loss(self, unique_item_idx, unique_item_llm_row, temperature):
        """Symmetric InfoNCE between the learned ID embedding and the
        projected frozen LLM embedding, for DISTINCT items only (caller
        must de-duplicate item_idx before calling — see train_epoch)."""
        z_id = F.normalize(self.item_emb(unique_item_idx), dim=-1)
        z_llm = F.normalize(self.item_llm_proj(self.item_llm_frozen(unique_item_llm_row)), dim=-1)
        logits = z_id @ z_llm.t() / temperature
        labels = torch.arange(z_id.size(0), device=z_id.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2


# ------------------------------------------------------------------
# Train / eval
# ------------------------------------------------------------------
def train_epoch(model, loader, optimizer, bce, lambda_align, temperature):
    model.train()
    total_loss = 0.0
    total_main = 0.0
    total_align = 0.0
    n = 0
    for u, item_idx, item_llm_row, click, purchase in loader:
        u = u.to(DEVICE); item_idx = item_idx.to(DEVICE); item_llm_row = item_llm_row.to(DEVICE)
        click = click.to(DEVICE); purchase = purchase.to(DEVICE)

        optimizer.zero_grad()
        p_ctr, _p_cvr, p_ctcvr = model(u, item_idx, item_llm_row)
        main_loss = bce(p_ctr, click) + bce(p_ctcvr, purchase)

        # de-duplicate items in this batch before computing alignment loss
        # (see docstring — avoids spurious false negatives from repeats)
        unique_idx = torch.unique(item_idx)
        unique_idx = unique_idx[unique_idx != 0]  # skip UNK, not a real item
        if len(unique_idx) > 1:
            idx_to_row = {}
            for a, b in zip(item_idx.tolist(), item_llm_row.tolist()):
                if a != 0 and a not in idx_to_row:
                    idx_to_row[a] = b
            unique_llm_rows = torch.tensor([idx_to_row[i.item()] for i in unique_idx],
                                            device=DEVICE, dtype=torch.long)
            align_loss = model.alignment_loss(unique_idx, unique_llm_rows, temperature)
        else:
            align_loss = torch.tensor(0.0, device=DEVICE)

        loss = main_loss + lambda_align * align_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(u)
        total_main += main_loss.item() * len(u)
        total_align += align_loss.item() * len(u)
        n += len(u)
    return total_loss / n, total_main / n, total_align / n


@torch.no_grad()
def predict(model, loader):
    model.eval()
    p_ctrs, p_cvrs, p_ctcvrs, clicks, purchases = [], [], [], [], []
    for u, item_idx, item_llm_row, click, purchase in loader:
        u, item_idx, item_llm_row = u.to(DEVICE), item_idx.to(DEVICE), item_llm_row.to(DEVICE)
        p_ctr, p_cvr, p_ctcvr = model(u, item_idx, item_llm_row)
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

    item_llm_data = build_item_llm_embeddings()
    item_id_to_llm_row = item_llm_data["item_id_to_llm_row"]
    item_llm_matrix = item_llm_data["embeddings"]
    llm_unk_row = item_llm_data["unk_row"]

    print("\nLoading train/val/test CSVs ...")
    t0 = time.time()
    train_df = load_csv(TRAIN_PATH)
    val_df = load_csv(VAL_PATH)
    test_df = load_csv(TEST_PATH)
    print(f"  train={len(train_df):,} val={len(val_df):,} test={len(test_df):,} "
          f"({time.time()-t0:.0f}s)")

    print("Building user/item vocab from TRAIN only (same as baseline) ...")
    user_vocab = build_vocab(train_df["user_id"])
    item_vocab = build_vocab(train_df["item_id"])
    n_users, n_items = len(user_vocab), len(item_vocab)
    print(f"  n_users={n_users:,} n_items={n_items:,}")

    train_t = make_tensors(train_df, user_vocab, item_vocab, item_id_to_llm_row, llm_unk_row)
    val_t = make_tensors(val_df, user_vocab, item_vocab, item_id_to_llm_row, llm_unk_row)
    test_t = make_tensors(test_df, user_vocab, item_vocab, item_id_to_llm_row, llm_unk_row)
    test_cold_mask = test_df["is_cold_start_item"].values.astype(bool)

    train_loader = DataLoader(TensorDataset(*train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_t), batch_size=BATCH_SIZE * 4, shuffle=False)
    test_loader = DataLoader(TensorDataset(*test_t), batch_size=BATCH_SIZE * 4, shuffle=False)

    print(f"\nDevice: {DEVICE}")
    model = ESMM_V2_Aligned(n_users, n_items, item_llm_matrix, USER_EMBED_DIM, ITEM_EMBED_DIM,
                             HIDDEN).to(DEVICE)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    bce = nn.BCELoss()

    best_val_ctcvr_auc = -1.0
    best_epoch = -1
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, main_loss, align_loss = train_epoch(
            model, train_loader, optimizer, bce, LAMBDA_ALIGN, CONTRASTIVE_TEMPERATURE)
        p_ctr, p_cvr, p_ctcvr, click, purchase = predict(model, val_loader)
        val_metrics = compute_metrics(p_ctr, p_cvr, p_ctcvr, click, purchase)
        print(f"Epoch {epoch:>2} | loss={train_loss:.5f} (main={main_loss:.5f}, "
              f"align={align_loss:.5f}) | "
              f"val CTR-AUC={val_metrics.get('ctr_auc', float('nan')):.4f} | "
              f"val CVR-AUC={val_metrics.get('cvr_auc_post_click', float('nan')):.4f} | "
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

    results = {"seed": seed, "best_epoch": best_epoch, "lm_name": LM_NAME, "lambda_align": LAMBDA_ALIGN,
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
    print("FINAL RESULTS (V2-MPNet — RLMRec-style ID + LLM alignment, MPNet encoder)")
    print("=" * 70)
    print(json.dumps(results, indent=2))

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    print(f"\nCompare against v2_results/esmm_v2_aligned_metrics.json (MiniLM) "
          f"to see whether encoder quality moved the numbers.")


if __name__ == "__main__":
    main()

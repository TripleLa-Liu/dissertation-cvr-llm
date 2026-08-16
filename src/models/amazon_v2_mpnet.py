"""
Amazon Reviews'23 (Video_Games) V2-MPNet — RLMRec-style alignment, MPNet encoder
============================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

"Different Embedders" experiment, extended to Amazon (2026-08-15 supervisor
request — the MiniLM-vs-MPNet comparison had only ever been run on Ali-CCP,
leaving Amazon's model-variant x embedder x dataset matrix incomplete).
Identical to amazon_v2_aligned.py (V2), with the frozen sentence-transformer
swapped from all-MiniLM-L6-v2 (384-dim) to all-mpnet-base-v2 (768-dim). V2
was chosen as the base for this comparison for the same reason as on
Ali-CCP (llm_encoder_v2_mpnet.py): it is the only Amazon LLM-integrated
variant competitive with the ID baseline, making it the more informative
test bed for "does encoder quality matter" than V1, which already
underperforms regardless of embedder.

Everything else (architecture, train/val/test split, alignment loss,
evaluation protocol) is unchanged from amazon_v2_aligned.py — see that
file for full design rationale. Writes to a separate v2_mpnet_results/
directory so this run doesn't overwrite the MiniLM-based V2 checkpoint.

Requires amazon_train.csv / amazon_val.csv / amazon_test.csv and
amazon_item_text.csv (all from amazon_build_dataset.py) in
WORK_DIR/amazon/processed/.
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
# Local Mac run (2026-08-15, Amazon MPNet gap) — see amazon_download.py.
WORK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "migration_package", "processed_data")
PROCESSED_DIR = os.path.join(WORK_DIR, "amazon", "processed")
RESULTS_DIR = os.path.join(WORK_DIR, "amazon", "v2_mpnet_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(PROCESSED_DIR, "amazon_train.csv")
VAL_PATH = os.path.join(PROCESSED_DIR, "amazon_val.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "amazon_test.csv")
ITEM_TEXT_PATH = os.path.join(PROCESSED_DIR, "amazon_item_text.csv")

# independent cache — separate from v2_aligned_results/ (MiniLM) so both
# can be compared side by side without either overwriting the other
ITEM_EMBED_CACHE = os.path.join(RESULTS_DIR, "item_llm_embeddings.pkl")
VOCAB_PATH = os.path.join(RESULTS_DIR, "id_vocab.pkl")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "amazon_v2_mpnet.pt")
METRICS_PATH = os.path.join(RESULTS_DIR, "amazon_v2_mpnet_metrics.json")

LM_NAME = "all-mpnet-base-v2"   # only change vs. amazon_v2_aligned.py (was all-MiniLM-L6-v2)
USER_EMBED_DIM = 32
ITEM_EMBED_DIM = 32
HIDDEN = [128, 64]
BATCH_SIZE = 1024
EPOCHS = 20
PATIENCE = 3
LR = 1e-3
WEIGHT_DECAY = 1e-6
LAMBDA_ALIGN = 0.1
CONTRASTIVE_TEMPERATURE = 0.1
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
# Frozen item LLM embeddings (same pattern as amazon_v2_aligned.py,
# including its stale-cache guard)
# ------------------------------------------------------------------
def build_item_llm_embeddings():
    print(f"Loading {ITEM_TEXT_PATH} ...")
    df = pd.read_csv(ITEM_TEXT_PATH, dtype={"item_id": str, "real_text": str})
    print(f"  {len(df):,} distinct items")

    if os.path.exists(ITEM_EMBED_CACHE):
        with open(ITEM_EMBED_CACHE, "rb") as f:
            cached = pickle.load(f)
        cached_ids = set(cached["item_id_to_llm_row"].keys())
        current_ids = set(df["item_id"])
        if cached_ids >= current_ids:
            print(f"Loading cached item LLM embeddings from {ITEM_EMBED_CACHE} ...")
            return cached
        else:
            print(f"Cached embeddings at {ITEM_EMBED_CACHE} don't cover the current "
                  f"item set ({len(cached_ids & current_ids):,}/{len(current_ids):,} "
                  f"items match) — re-encoding instead of trusting it.")

    print(f"Loading sentence-transformer '{LM_NAME}' (frozen) ...")
    model = SentenceTransformer(LM_NAME, device=str(DEVICE))
    print("Encoding item text ...")
    t0 = time.time()
    embeddings = model.encode(df["real_text"].fillna("").tolist(), batch_size=256,
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


def make_tensors(df, user_vocab, item_vocab, item_id_to_llm_row, llm_unk_row):
    u = encode(df["user_id"], user_vocab)
    i_idx = encode(df["item_id"], item_vocab)  # 0 = unseen in train
    i_llm_row = df["item_id"].map(lambda x: item_id_to_llm_row.get(x, llm_unk_row)).astype("int64").values.copy()
    label = df["label"].values.astype("float32")
    return (torch.from_numpy(u), torch.from_numpy(i_idx), torch.from_numpy(i_llm_row),
            torch.from_numpy(label))


# ------------------------------------------------------------------
# Model (identical to V2Aligned in amazon_v2_aligned.py — lm_dim is read
# dynamically from the embedding matrix, so this class needs no changes
# for the larger 768-dim MPNet vectors)
# ------------------------------------------------------------------
class V2Aligned(nn.Module):
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

        d = user_dim + item_dim
        layers = []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.tower = nn.Sequential(*layers)

    def item_representation(self, item_idx, item_llm_row):
        """idx != 0 (seen in train) -> learned ID embedding.
        idx == 0 (cold-start, unseen in train) -> projected frozen LLM embedding."""
        id_repr = self.item_emb(item_idx)
        llm_repr = self.item_llm_proj(self.item_llm_frozen(item_llm_row))
        use_id = (item_idx != 0).float().unsqueeze(-1)
        return use_id * id_repr + (1 - use_id) * llm_repr

    def forward(self, u, item_idx, item_llm_row):
        item_repr = self.item_representation(item_idx, item_llm_row)
        x = torch.cat([self.user_emb(u), item_repr], dim=1)
        return torch.sigmoid(self.tower(x)).squeeze(-1)

    def alignment_loss(self, unique_item_idx, unique_item_llm_row, temperature):
        """Symmetric InfoNCE between the learned ID embedding and the
        projected frozen LLM embedding, for DISTINCT items only (caller
        must de-duplicate item_idx before calling)."""
        z_id = F.normalize(self.item_emb(unique_item_idx), dim=-1)
        z_llm = F.normalize(self.item_llm_proj(self.item_llm_frozen(unique_item_llm_row)), dim=-1)
        logits = z_id @ z_llm.t() / temperature
        labels = torch.arange(z_id.size(0), device=z_id.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2


# ------------------------------------------------------------------
# Train / eval (identical logic to amazon_v2_aligned.py)
# ------------------------------------------------------------------
def train_epoch(model, loader, optimizer, bce, lambda_align, temperature):
    model.train()
    total_loss = total_main = total_align = 0.0
    n = 0
    for u, item_idx, item_llm_row, label in loader:
        u, item_idx, item_llm_row, label = (
            u.to(DEVICE), item_idx.to(DEVICE), item_llm_row.to(DEVICE), label.to(DEVICE))

        optimizer.zero_grad()
        p = model(u, item_idx, item_llm_row)
        main_loss = bce(p, label)

        unique_idx = torch.unique(item_idx)
        unique_idx = unique_idx[unique_idx != 0]
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
    preds, labels = [], []
    for u, item_idx, item_llm_row, label in loader:
        u, item_idx, item_llm_row = u.to(DEVICE), item_idx.to(DEVICE), item_llm_row.to(DEVICE)
        p = model(u, item_idx, item_llm_row)
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

    print("Building user/item vocab from TRAIN only ...")
    user_vocab = build_vocab(train_df["user_id"])
    item_vocab = build_vocab(train_df["item_id"])
    n_users, n_items = len(user_vocab), len(item_vocab)
    print(f"  n_users={n_users:,} n_items={n_items:,}")
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump({"user_vocab": user_vocab, "item_vocab": item_vocab}, f)

    train_t = make_tensors(train_df, user_vocab, item_vocab, item_id_to_llm_row, llm_unk_row)
    val_t = make_tensors(val_df, user_vocab, item_vocab, item_id_to_llm_row, llm_unk_row)
    test_t = make_tensors(test_df, user_vocab, item_vocab, item_id_to_llm_row, llm_unk_row)
    test_cold_mask = test_df["is_cold_start_item"].values.astype(bool)

    train_loader = DataLoader(TensorDataset(*train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_t), batch_size=BATCH_SIZE * 4, shuffle=False)
    test_loader = DataLoader(TensorDataset(*test_t), batch_size=BATCH_SIZE * 4, shuffle=False)

    print(f"\nDevice: {DEVICE}")
    model = V2Aligned(n_users, n_items, item_llm_matrix, USER_EMBED_DIM, ITEM_EMBED_DIM,
                       HIDDEN).to(DEVICE)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    bce = nn.BCELoss()

    best_val_auc = -1.0
    best_epoch = -1
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, main_loss, align_loss = train_epoch(
            model, train_loader, optimizer, bce, LAMBDA_ALIGN, CONTRASTIVE_TEMPERATURE)
        preds, labels = predict(model, val_loader)
        val_metrics = compute_metrics(preds, labels)
        print(f"Epoch {epoch:>2} | loss={train_loss:.5f} (main={main_loss:.5f}, "
              f"align={align_loss:.5f}) | val AUC={val_metrics.get('auc', float('nan')):.4f} | "
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

    results = {"seed": seed, "best_epoch": best_epoch, "lm_name": LM_NAME,
               "lambda_align": LAMBDA_ALIGN, "n_users_train_vocab": n_users,
               "n_items_train_vocab": n_items}

    preds, labels = predict(model, val_loader)
    results["val"] = compute_metrics(preds, labels)

    preds, labels = predict(model, test_loader)
    results["test_overall"] = compute_metrics(preds, labels)
    results["test_seen_items"] = compute_metrics(preds, labels, mask=~test_cold_mask)
    results["test_cold_start_items"] = compute_metrics(preds, labels, mask=test_cold_mask)

    print("\n" + "=" * 70)
    print("FINAL RESULTS (Amazon Video_Games — V2-MPNet RLMRec-style ID + LLM alignment)")
    print("=" * 70)
    print(json.dumps(results, indent=2))

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    print(f"\nCompare against v2_aligned_results/amazon_v2_aligned_metrics.json (MiniLM) "
          f"to see whether encoder quality moved the numbers.")


if __name__ == "__main__":
    main()

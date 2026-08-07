"""
Amazon Reviews'23 (Video_Games) V4 — LLM-Aligned Dynamic Graph (DGSR-lite)
============================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

First model in this dissertation that actually implements graph/sequential
structure — Baseline/V1/V1-Full/V2/V3 are all static two-tower models that
differ only in item representation (ID vs frozen-LLM-text), none of them
aggregate over a user's interaction history. This is the "Dynamic Graph"
layer of the three-layer architecture (LLM Encoder + Dynamic Graph +
Delayed Feedback Correction), Amazon-only because Ali-CCP has no
per-interaction timestamps (see README).

Deliberate scope reduction vs. DGSR (Zhang et al., TKDE 2022), documented
here rather than silently simplified:
  - DGSR: edge quintuple (u, i, t, o_u, o_i) + dual-channel long/short-term
    attention, evolving BOTH user and item representations.
  - This (V4): single-channel target-attention (query = target item,
    keys/values = user's prior interacted items) + a recency-decay bias
    from real Delta-t, in the spirit of DIN's target-attention (Zhou et
    al., KDD 2018, already cited) rather than DGSR's dual channels. Only
    the USER side is made dynamic; item-side history aggregation is left
    as future work. No explicit order-position feature (o_u/o_i) —
    Delta-t (real elapsed time, which Ali-CCP could never provide) is used
    directly instead.
  Chosen given the ~1-month submission timeline: lower implementation
  risk, and still directly tests the dissertation's actual title claim
  ("Dynamic Graph Networks") which nothing prior to this script does.

Combines the "Dynamic Graph" layer with the "LLM Encoder" layer from the
start (per 2026-08-07 planning discussion: a rough combined result that
validates the full architecture is worth more than a polished ID-only
ablation given the time budget) — reuses amazon_v2_aligned.py's ID+LLM
alignment pattern (item_emb for train-vocab items, item_llm_proj(frozen
MiniLM) for cold-start items, InfoNCE alignment loss) for BOTH the target
item and every history item, then adds the temporal attention aggregator
on top. An ID-only ablation (drop item_llm_proj, use item_emb alone) is a
natural follow-up once this combined version is validated, not done here.

Requires (all from existing pipeline, no new raw-data dependency):
  - amazon_train.csv / amazon_val.csv / amazon_test.csv (from
    amazon_build_dataset.py, 2026-08-07+ version — must have a
    `timestamp` column)
  - amazon_item_text.csv (from amazon_build_dataset.py)
  - amazon_user_histories.pkl (from amazon_build_user_histories.py)

Cold-start / no-history fallback: a row whose user has zero prior
positive interactions before its own timestamp (18.9% of test rows, see
amazon_build_user_histories.py's diagnostic output) gets an all-zero
dynamic_user_repr (attention over an empty/fully-masked history) and
falls back to user_emb + target item representation alone — same
UNK-style convention used throughout this codebase for missing signal.
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
PROCESSED_DIR = os.path.join(WORK_DIR, "amazon", "processed")
RESULTS_DIR = os.path.join(WORK_DIR, "amazon", "v4_dynamic_graph_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(PROCESSED_DIR, "amazon_train.csv")
VAL_PATH = os.path.join(PROCESSED_DIR, "amazon_val.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "amazon_test.csv")
ITEM_TEXT_PATH = os.path.join(PROCESSED_DIR, "amazon_item_text.csv")
HISTORIES_PATH = os.path.join(PROCESSED_DIR, "amazon_user_histories.pkl")

# independent cache — separate from v2_aligned_results/ / text_embedding_results/
ITEM_EMBED_CACHE = os.path.join(RESULTS_DIR, "item_llm_embeddings.pkl")
VOCAB_PATH = os.path.join(RESULTS_DIR, "id_vocab.pkl")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "amazon_dynamic_graph_v4.pt")
METRICS_PATH = os.path.join(RESULTS_DIR, "amazon_dynamic_graph_v4_metrics.json")

LM_NAME = "all-MiniLM-L6-v2"   # same encoder as V1/V2, for a fair comparison
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
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED, help="random seed (default: 42)")
    return p.parse_args()


def seed_suffixed(path, seed):
    if seed == SEED:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_seed{seed}{ext}"


# ------------------------------------------------------------------
# Frozen item LLM embeddings (identical pattern to amazon_v2_aligned.py)
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
                  f"item set — re-encoding instead of trusting it.")

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


def make_target_tensors(df, user_vocab, item_vocab, item_id_to_llm_row, llm_unk_row):
    u = encode(df["user_id"], user_vocab)
    i_idx = encode(df["item_id"], item_vocab)
    i_llm_row = df["item_id"].map(lambda x: item_id_to_llm_row.get(x, llm_unk_row)).astype("int64").values.copy()
    label = df["label"].values.astype("float32")
    return u, i_idx, i_llm_row, label


# ------------------------------------------------------------------
# Causal history tensors — see amazon_build_user_histories.py for how
# `histories` is built (per-user sorted (item_id, timestamp), positives
# only). This does the per-row bisect lookup + truncate-to-most-recent-N
# + pad, producing fixed-shape (n_rows, max_hist) arrays the model can
# batch over. Pad slots reuse idx=0 / llm_unk_row (existing UNK
# convention) — safe because the attention layer masks them out
# explicitly, so their embedded VALUE never actually reaches the output.
# ------------------------------------------------------------------
def compute_history_tensors(df, histories, item_vocab, item_id_to_llm_row, llm_unk_row, max_hist):
    n = len(df)
    hist_idx = np.zeros((n, max_hist), dtype="int64")
    hist_llm_row = np.full((n, max_hist), llm_unk_row, dtype="int64")
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
        k = len(sel_items)
        for j in range(k):
            iid = sel_items[j]
            hist_idx[row_i, j] = item_vocab.get(iid, 0)
            hist_llm_row[row_i, j] = item_id_to_llm_row.get(iid, llm_unk_row)
            hist_delta_days[row_i, j] = max(0.0, (t - sel_ts[j]) / 86400.0)
            hist_mask[row_i, j] = True

    print(f"    {n_with_history:,}/{n:,} rows ({n_with_history/n:.1%}) have >=1 prior history item")
    return hist_idx, hist_llm_row, hist_delta_days, hist_mask


def make_tensors(df, user_vocab, item_vocab, item_id_to_llm_row, llm_unk_row, histories, max_hist):
    u, i_idx, i_llm_row, label = make_target_tensors(df, user_vocab, item_vocab, item_id_to_llm_row, llm_unk_row)
    hist_idx, hist_llm_row, hist_delta_days, hist_mask = compute_history_tensors(
        df, histories, item_vocab, item_id_to_llm_row, llm_unk_row, max_hist)
    return (torch.from_numpy(u), torch.from_numpy(i_idx), torch.from_numpy(i_llm_row),
            torch.from_numpy(hist_idx), torch.from_numpy(hist_llm_row),
            torch.from_numpy(hist_delta_days), torch.from_numpy(hist_mask),
            torch.from_numpy(label))


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
class TemporalAttention(nn.Module):
    """Single-channel target-attention over a user's history, biased by
    real elapsed time (Delta-t in days). `raw_decay` is softplus'd to stay
    positive — larger Delta-t always pulls the attention logit down (more
    recent interactions preferred), the model only learns HOW MUCH to
    prefer recency, not the sign. Rows with zero valid history (mask all
    False) get an explicit zero output rather than going through softmax
    (avoids NaN from an all -inf row)."""

    def __init__(self):
        super().__init__()
        self.raw_decay = nn.Parameter(torch.tensor(0.1))

    def forward(self, query, keys, values, delta_days, mask):
        # query: (B, d)  keys/values: (B, N, d)  delta_days/mask: (B, N)
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


class DynamicGraphV4(nn.Module):
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
        self.temporal_attn = TemporalAttention()

        d = user_dim + item_dim + item_dim  # user_emb + dynamic_user_repr + target item_repr
        layers = []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.tower = nn.Sequential(*layers)

    def item_representation(self, item_idx, item_llm_row):
        """Works for both (B,) target items and (B,N) history items —
        nn.Linear/nn.Embedding both broadcast over arbitrary leading dims,
        so no separate batched variant is needed. idx==0 (unseen in train,
        OR a padding slot) -> projected frozen LLM embedding branch;
        padding slots additionally get masked out entirely in
        TemporalAttention, so their exact value here doesn't matter."""
        id_repr = self.item_emb(item_idx)
        llm_repr = self.item_llm_proj(self.item_llm_frozen(item_llm_row))
        use_id = (item_idx != 0).float().unsqueeze(-1)
        return use_id * id_repr + (1 - use_id) * llm_repr

    def forward(self, u, item_idx, item_llm_row, hist_idx, hist_llm_row, hist_delta_days, hist_mask):
        target_repr = self.item_representation(item_idx, item_llm_row)          # (B, item_dim)
        hist_repr = self.item_representation(hist_idx, hist_llm_row)            # (B, N, item_dim)
        dynamic_user_repr = self.temporal_attn(target_repr, hist_repr, hist_repr, hist_delta_days, hist_mask)
        x = torch.cat([self.user_emb(u), dynamic_user_repr, target_repr], dim=1)
        return torch.sigmoid(self.tower(x)).squeeze(-1)

    def alignment_loss(self, unique_item_idx, unique_item_llm_row, temperature):
        """Same symmetric InfoNCE as amazon_v2_aligned.py, unchanged."""
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
    total_loss = total_main = total_align = 0.0
    n = 0
    for u, item_idx, item_llm_row, hist_idx, hist_llm_row, hist_delta_days, hist_mask, label in loader:
        u, item_idx, item_llm_row = u.to(DEVICE), item_idx.to(DEVICE), item_llm_row.to(DEVICE)
        hist_idx, hist_llm_row = hist_idx.to(DEVICE), hist_llm_row.to(DEVICE)
        hist_delta_days, hist_mask, label = hist_delta_days.to(DEVICE), hist_mask.to(DEVICE), label.to(DEVICE)

        optimizer.zero_grad()
        p = model(u, item_idx, item_llm_row, hist_idx, hist_llm_row, hist_delta_days, hist_mask)
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
    for u, item_idx, item_llm_row, hist_idx, hist_llm_row, hist_delta_days, hist_mask, label in loader:
        u, item_idx, item_llm_row = u.to(DEVICE), item_idx.to(DEVICE), item_llm_row.to(DEVICE)
        hist_idx, hist_llm_row = hist_idx.to(DEVICE), hist_llm_row.to(DEVICE)
        hist_delta_days, hist_mask = hist_delta_days.to(DEVICE), hist_mask.to(DEVICE)
        p = model(u, item_idx, item_llm_row, hist_idx, hist_llm_row, hist_delta_days, hist_mask)
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
            f"data/preprocessing/amazon_build_user_histories.py first "
            f"(requires amazon_train/val/test.csv to have a timestamp column, "
            f"i.e. amazon_build_dataset.py rerun after the 2026-08-07 update).")

    item_llm_data = build_item_llm_embeddings()
    item_id_to_llm_row = item_llm_data["item_id_to_llm_row"]
    item_llm_matrix = item_llm_data["embeddings"]
    llm_unk_row = item_llm_data["unk_row"]

    print(f"\nLoading {HISTORIES_PATH} ...")
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
        raise SystemExit(
            "amazon_train.csv has no `timestamp` column — rerun amazon_build_dataset.py "
            "(2026-08-07+ version) before this script.")
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
    train_t = make_tensors(train_df, user_vocab, item_vocab, item_id_to_llm_row, llm_unk_row, histories, max_hist)
    print("  val:")
    val_t = make_tensors(val_df, user_vocab, item_vocab, item_id_to_llm_row, llm_unk_row, histories, max_hist)
    print("  test:")
    test_t = make_tensors(test_df, user_vocab, item_vocab, item_id_to_llm_row, llm_unk_row, histories, max_hist)
    print(f"  done ({time.time()-t0:.0f}s)")
    test_cold_mask = test_df["is_cold_start_item"].values.astype(bool)

    train_loader = DataLoader(TensorDataset(*train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_t), batch_size=BATCH_SIZE * 4, shuffle=False)
    test_loader = DataLoader(TensorDataset(*test_t), batch_size=BATCH_SIZE * 4, shuffle=False)

    print(f"\nDevice: {DEVICE}")
    model = DynamicGraphV4(n_users, n_items, item_llm_matrix, USER_EMBED_DIM, ITEM_EMBED_DIM,
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

    results = {"seed": seed, "best_epoch": best_epoch, "lm_name": LM_NAME,
               "lambda_align": LAMBDA_ALIGN, "max_history": max_hist,
               "n_users_train_vocab": n_users, "n_items_train_vocab": n_items,
               "learned_decay_rate": F.softplus(model.temporal_attn.raw_decay).item()}

    preds, labels = predict(model, val_loader)
    results["val"] = compute_metrics(preds, labels)

    preds, labels = predict(model, test_loader)
    results["test_overall"] = compute_metrics(preds, labels)
    results["test_seen_items"] = compute_metrics(preds, labels, mask=~test_cold_mask)
    results["test_cold_start_items"] = compute_metrics(preds, labels, mask=test_cold_mask)

    print("\n" + "=" * 70)
    print("FINAL RESULTS (Amazon Video_Games — V4 LLM-Aligned Dynamic Graph)")
    print("=" * 70)
    print(json.dumps(results, indent=2))

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    print(f"\nCompare against baseline_results/, text_embedding_results/ (V1-equiv), "
          f"v2_aligned_results/ (V2), and v3_hybrid_results.json.")


if __name__ == "__main__":
    main()

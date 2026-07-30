"""
Ali-CCP LLM Encoder V1 (FULL) — frozen small-LM item AND user embeddings
============================================================================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

WHAT THIS IS
-------------
Completes V1 per the supervisor meeting todo ("extract full categorical
feature set... add LLM embedding"). llm_encoder_v1.py only replaced the
ITEM side with LLM text; user side was still a learned ID embedding. This
version replaces BOTH sides with frozen `all-MiniLM-L6-v2` embeddings of
template pseudo-text (item: category/shop/intention-node/brand from
extract_item_pseudo_text.py; user: gender/age/spending-power/shopping-
depth/occupation/city-tier/segment from extract_user_pseudo_text.py) +
separate trainable adapters, feeding the same ESMM two-tower architecture.

WHY THIS ALSO MATTERS BEYOND "COMPLETING V1"
------------------------------------------------
Baseline #1's train vocabulary covered only 9,074 users (vs 140,092
items) — flagged as a likely driver of the val->test generalisation gap,
since unseen test users get the same zero-vector UNK problem unseen items
do. user_pseudo_text.csv covers every user_id across train/val/test (not
just train — see extract_user_pseudo_text.py), so this version has NO
user-side UNK problem either, same as the item side already didn't in
llm_encoder_v1.py. If the user-side generalisation gap hypothesis is
right, this full version should show a smaller val->test drop than both
baseline #1 and the item-only V1.

CAVEAT — same as llm_encoder_v1.py and README's "LLM text feasibility"
note: all pseudo-text is built from anonymised numeric feat_ids with no
public decoder. This is still testing "does an LM's architecture handle
these IDs better than a from-scratch embedding table", not "does genuine
pretrained world knowledge help" (see llm_encoder_v1.py's results — CTR
improved on cold-start, CVR got worse — the same dynamic may or may not
repeat here).

USAGE
-----
1. Run extract_user_pseudo_text.py first if user_pseudo_text.csv doesn't
   exist yet (item_pseudo_text.csv should already exist from
   llm_encoder_v1.py's run).
2. Run: python llm_encoder_v1_full.py
3. Send me the printed FINAL RESULTS — three-way comparison against
   baseline #1 and llm_encoder_v1.py (item-only), especially val->test
   generalisation gap and the seen/cold-start breakdown.
"""
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
WORK_DIR = r"E:\BaiduNetdiskDownload\Dataset\_processed"
RESULTS_DIR = r"E:\BaiduNetdiskDownload\Dataset\_processed\v1_full_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(WORK_DIR, "aliccp_train_split.csv")
VAL_PATH = os.path.join(WORK_DIR, "aliccp_val_split.csv")
TEST_PATH = os.path.join(WORK_DIR, "aliccp_test_filtered_joined.csv")
ITEM_TEXT_PATH = os.path.join(WORK_DIR, "item_pseudo_text.csv")
USER_TEXT_PATH = os.path.join(WORK_DIR, "user_pseudo_text.csv")

# NOTE: llm_encoder_v1.py's cache uses a different dict schema (key
# "item_id_to_row" vs this script's "id_to_row") — pointing at its own
# cache file here rather than reusing v1_results/item_llm_embeddings.pkl,
# to avoid a schema mismatch (hit a KeyError this way once already,
# 2026-07-21). Costs one extra re-encode of item text (a minute or two on
# GPU) but keeps the two scripts independent and robust.
ITEM_EMBED_CACHE = os.path.join(RESULTS_DIR, "item_llm_embeddings.pkl")
USER_EMBED_CACHE = os.path.join(RESULTS_DIR, "user_llm_embeddings.pkl")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "esmm_llm_v1_full.pt")
METRICS_PATH = os.path.join(RESULTS_DIR, "esmm_llm_v1_full_metrics.json")

LM_NAME = "all-MiniLM-L6-v2"
ADAPTER_DIM = 32
HIDDEN = [128, 64]
BATCH_SIZE = 4096
EPOCHS = 15
PATIENCE = 3
LR = 1e-3
WEIGHT_DECAY = 1e-6
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# Frozen LLM embeddings for items / users (precomputed once, cached)
# ------------------------------------------------------------------
def build_embeddings(text_path, id_col, cache_path, model=None):
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path} ...")
        with open(cache_path, "rb") as f:
            return pickle.load(f), model

    print(f"Loading {text_path} ...")
    df = pd.read_csv(text_path, dtype={id_col: str, "pseudo_text": str})
    print(f"  {len(df):,} distinct {id_col} values")

    if model is None:
        print(f"Loading sentence-transformer '{LM_NAME}' (frozen, no fine-tuning) ...")
        model = SentenceTransformer(LM_NAME, device=str(DEVICE))

    print(f"Encoding pseudo-text for all {id_col} values (batched) ...")
    t0 = time.time()
    texts = df["pseudo_text"].tolist()
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True,
                               convert_to_numpy=True).astype("float32")
    print(f"  Encoded {len(texts):,} in {time.time()-t0:.0f}s, dim={embeddings.shape[1]}")

    id_to_row = {v: i for i, v in enumerate(df[id_col])}
    # explicit UNK row (zeros) — see llm_encoder_v1.py's bugfix note: never
    # use -1 as a fallback index for nn.Embedding lookups.
    embeddings = np.vstack([embeddings, np.zeros((1, embeddings.shape[1]), dtype="float32")])
    unk_row = embeddings.shape[0] - 1
    result = {"id_to_row": id_to_row, "embeddings": embeddings, "unk_row": unk_row}
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Cached to {cache_path}")
    return result, model


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def load_csv(path):
    df = pd.read_csv(path, dtype={"user_id": str, "item_id": str,
                                   "click": "int8", "purchase": "int8"})
    df["user_id"] = df["user_id"].fillna("__MISSING__")
    return df


def encode_rows(series, id_to_row, unk_row, label):
    rows = series.map(lambda x: id_to_row.get(x, unk_row)).astype("int64").values
    n_missing = int((rows == unk_row).sum())
    if n_missing:
        print(f"  WARNING: {n_missing} rows have a {label} missing from its pseudo-text file "
              f"— falling back to the zero-vector UNK row.")
    return rows


def make_tensors(df, item_id_to_row, item_unk, user_id_to_row, user_unk):
    i_row = encode_rows(df["item_id"], item_id_to_row, item_unk, "item_id")
    u_row = encode_rows(df["user_id"], user_id_to_row, user_unk, "user_id")
    click = df["click"].values.astype("float32")
    purchase = df["purchase"].values.astype("float32")
    return (torch.from_numpy(u_row), torch.from_numpy(i_row),
            torch.from_numpy(click), torch.from_numpy(purchase))


# ------------------------------------------------------------------
# Model — both sides frozen LLM embeddings + adapters
# ------------------------------------------------------------------
class ESMM_LLM_V1_Full(nn.Module):
    def __init__(self, user_embed_matrix, item_embed_matrix, adapter_dim=32, hidden=(128, 64)):
        super().__init__()
        self.user_emb_frozen = nn.Embedding.from_pretrained(
            torch.from_numpy(user_embed_matrix), freeze=True)
        self.item_emb_frozen = nn.Embedding.from_pretrained(
            torch.from_numpy(item_embed_matrix), freeze=True)

        user_lm_dim = user_embed_matrix.shape[1]
        item_lm_dim = item_embed_matrix.shape[1]
        self.user_adapter = self._make_adapter(user_lm_dim, adapter_dim)
        self.item_adapter = self._make_adapter(item_lm_dim, adapter_dim)

        tower_in = adapter_dim * 2
        self.ctr_tower = self._make_tower(tower_in, hidden)
        self.cvr_tower = self._make_tower(tower_in, hidden)

    @staticmethod
    def _make_adapter(in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, out_dim),
        )

    @staticmethod
    def _make_tower(in_dim, hidden):
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1)]
            d = h
        layers += [nn.Linear(d, 1)]
        return nn.Sequential(*layers)

    def forward(self, u_row, i_row):
        user_repr = self.user_adapter(self.user_emb_frozen(u_row))
        item_repr = self.item_adapter(self.item_emb_frozen(i_row))
        x = torch.cat([user_repr, item_repr], dim=1)
        p_ctr = torch.sigmoid(self.ctr_tower(x)).squeeze(-1)
        p_cvr = torch.sigmoid(self.cvr_tower(x)).squeeze(-1)
        p_ctcvr = p_ctr * p_cvr
        return p_ctr, p_cvr, p_ctcvr


# ------------------------------------------------------------------
# Train / eval (same logic as baseline / v1)
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

    item_data, lm_model = build_embeddings(ITEM_TEXT_PATH, "item_id", ITEM_EMBED_CACHE)
    user_data, _ = build_embeddings(USER_TEXT_PATH, "user_id", USER_EMBED_CACHE, model=lm_model)

    print("\nLoading train/val/test CSVs ...")
    t0 = time.time()
    train_df = load_csv(TRAIN_PATH)
    val_df = load_csv(VAL_PATH)
    test_df = load_csv(TEST_PATH)
    print(f"  train={len(train_df):,} val={len(val_df):,} test={len(test_df):,} "
          f"({time.time()-t0:.0f}s)")

    train_t = make_tensors(train_df, item_data["id_to_row"], item_data["unk_row"],
                            user_data["id_to_row"], user_data["unk_row"])
    val_t = make_tensors(val_df, item_data["id_to_row"], item_data["unk_row"],
                          user_data["id_to_row"], user_data["unk_row"])
    test_t = make_tensors(test_df, item_data["id_to_row"], item_data["unk_row"],
                           user_data["id_to_row"], user_data["unk_row"])
    test_cold_mask = test_df["is_cold_start_item"].values.astype(bool)

    train_loader = DataLoader(TensorDataset(*train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_t), batch_size=BATCH_SIZE * 4, shuffle=False)
    test_loader = DataLoader(TensorDataset(*test_t), batch_size=BATCH_SIZE * 4, shuffle=False)

    print(f"\nDevice: {DEVICE}")
    model = ESMM_LLM_V1_Full(user_data["embeddings"], item_data["embeddings"],
                              ADAPTER_DIM, HIDDEN).to(DEVICE)
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
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping. Best epoch: {best_epoch}")
                break

    print(f"\nLoading best checkpoint (epoch {best_epoch}) for final evaluation ...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH))

    results = {"best_epoch": best_epoch, "lm_name": LM_NAME, "adapter_dim": ADAPTER_DIM,
               "hidden": HIDDEN}

    p_ctr, p_cvr, p_ctcvr, click, purchase = predict(model, val_loader)
    results["val"] = compute_metrics(p_ctr, p_cvr, p_ctcvr, click, purchase)

    p_ctr, p_cvr, p_ctcvr, click, purchase = predict(model, test_loader)
    results["test_overall"] = compute_metrics(p_ctr, p_cvr, p_ctcvr, click, purchase)
    results["test_seen_items"] = compute_metrics(p_ctr, p_cvr, p_ctcvr, click, purchase,
                                                   mask=~test_cold_mask)
    results["test_cold_start_items"] = compute_metrics(p_ctr, p_cvr, p_ctcvr, click, purchase,
                                                          mask=test_cold_mask)

    print("\n" + "=" * 70)
    print("FINAL RESULTS (LLM Encoder V1 FULL — item + user frozen MiniLM text)")
    print("=" * 70)
    print(json.dumps(results, indent=2))

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {METRICS_PATH}")
    print("\nSend me this FINAL RESULTS block — three-way comparison against "
          "baseline #1 and the item-only V1 (both already in README).")


if __name__ == "__main__":
    main()

"""
Dataset EDA Script
==================
Dissertation: LLM-Enhanced Dynamic Graph Networks for CVR Prediction
Author: Liu Yize | UCL MSc KIDS

Covers:
  - Criteo Sponsored Search Conversion Log
  - Ali-CCP (Alibaba Click and Conversion Prediction)

Run one section at a time depending on which dataset you have downloaded.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["font.size"] = 11

# ============================================================
#  SECTION 1 — CRITEO
#  Download: http://go.criteo.net/criteo-research-search-conversion.tar.gz
#  (~310 MB compressed, ~6.4 GB uncompressed, tab-separated)
# ============================================================

CRITEO_PATH = "CriteoSearchData"   # <-- update to your file path

def load_criteo(path=CRITEO_PATH, nrows=500_000):
    """Load a sample of the Criteo dataset."""
    print(f"Loading Criteo ({nrows:,} rows)...")
    cols = [
        "Sale", "SalesAmountInEuro", "time_delay_for_conversion",
        "click_timestamp", "nb_clicks_1week", "product_price",
        "product_age_group", "device_type", "audience_id",
        "product_gender", "product_brand",
        "product_category_1", "product_category_2", "product_category_3",
        "product_category_4", "product_category_5", "product_category_6",
        "product_category_7",
        "product_country", "product_id", "product_title",
        "partner_id", "user_id"
    ]
    df = pd.read_csv(path, sep="\t", header=None, names=cols,
                     nrows=nrows, na_values="-1")
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def eda_criteo(df):
    print("\n" + "="*60)
    print("CRITEO — Exploratory Data Analysis")
    print("="*60)

    # ── 1. Basic info ──────────────────────────────────────────
    print("\n[1] Shape:", df.shape)
    print("\n[2] Dtypes:\n", df.dtypes)
    print("\n[3] Missing values (%):\n",
          (df.isnull().mean() * 100).round(2).sort_values(ascending=False))

    # ── 2. Label distribution ──────────────────────────────────
    cvr = df["Sale"].mean()
    print(f"\n[4] Conversion Rate: {cvr:.4%}")
    print(f"    Conversions: {df['Sale'].sum():,} / {len(df):,}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(["No Sale (0)", "Sale (1)"],
                df["Sale"].value_counts().sort_index(),
                color=["#EF9A9A", "#A5D6A7"])
    axes[0].set_title("Conversion Label Distribution")
    axes[0].set_ylabel("Count")
    for bar in axes[0].patches:
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 500,
                     f"{bar.get_height():,.0f}", ha="center", fontsize=9)

    # ── 3. Conversion delay ────────────────────────────────────
    delay = df.loc[df["Sale"] == 1, "time_delay_for_conversion"].dropna()
    delay_hours = delay / 3600

    axes[1].hist(delay_hours.clip(upper=delay_hours.quantile(0.99)),
                 bins=50, color="#90CAF9", edgecolor="white")
    axes[1].set_title("Conversion Delay Distribution (hours)")
    axes[1].set_xlabel("Hours from click to conversion")
    axes[1].set_ylabel("Count")

    # Cumulative delay
    sorted_delay = np.sort(delay_hours)
    cdf = np.arange(1, len(sorted_delay)+1) / len(sorted_delay)
    axes[2].plot(sorted_delay.clip(max=sorted_delay.quantile(0.99)), cdf,
                 color="#6C63FF", linewidth=2)
    axes[2].axvline(24, color="red", linestyle="--", label="24h")
    axes[2].axvline(72, color="orange", linestyle="--", label="72h")
    axes[2].set_title("CDF of Conversion Delay")
    axes[2].set_xlabel("Hours from click to conversion")
    axes[2].set_ylabel("Cumulative proportion")
    axes[2].legend()
    axes[2].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    plt.suptitle("Criteo Dataset — Key Statistics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("criteo_eda.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── 4. Delay statistics ────────────────────────────────────
    print(f"\n[5] Conversion delay stats (hours):")
    print(f"    Median : {delay_hours.median():.1f}h")
    print(f"    Mean   : {delay_hours.mean():.1f}h")
    print(f"    < 1h   : {(delay_hours < 1).mean():.1%}")
    print(f"    < 24h  : {(delay_hours < 24).mean():.1%}")
    print(f"    < 72h  : {(delay_hours < 72).mean():.1%}")
    print(f"    Max    : {delay_hours.max():.1f}h")

    # ── 5. Product features ────────────────────────────────────
    print("\n[6] Product price (non-missing):")
    price = df["product_price"].dropna()
    print(f"    Min={price.min():.2f}, Median={price.median():.2f}, "
          f"Mean={price.mean():.2f}, Max={price.max():.2f}")

    print("\n[7] Device type distribution:")
    print(df["device_type"].value_counts(normalize=True).mul(100).round(2))

    print("\n[8] Product gender distribution:")
    print(df["product_gender"].value_counts(normalize=True).mul(100).round(2))

    # ── 6. Temporal pattern ────────────────────────────────────
    df["hour"] = pd.to_datetime(df["click_timestamp"], unit="s").dt.hour
    hourly_cvr = df.groupby("hour")["Sale"].mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    hourly_cvr.plot(ax=ax, marker="o", color="#6C63FF", linewidth=2)
    ax.set_title("Criteo — Conversion Rate by Hour of Day")
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Conversion Rate")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("criteo_hourly_cvr.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n✅ Criteo EDA complete. Charts saved.")


# ============================================================
#  SECTION 2 — ALI-CCP
#  Download: https://tianchi.aliyun.com/dataset/408
#  Requires Alibaba Tianchi account (free registration)
#  Files after download:
#    - sample_skeleton_train.csv  (training set with click+conversion labels)
#    - sample_skeleton_test.csv   (test set)
#    - common_features_train.csv  (user & item feature store)
# ============================================================

ALICCP_TRAIN  = "sample_skeleton_train.csv"    # <-- update path
ALICCP_FEATS  = "common_features_train.csv"    # <-- update path

def load_aliccp(train_path=ALICCP_TRAIN, feat_path=ALICCP_FEATS, nrows=500_000):
    """Load Ali-CCP training set and feature store."""
    print(f"Loading Ali-CCP ({nrows:,} rows)...")

    # Main interaction log: sample_id | click | purchase | features...
    train = pd.read_csv(train_path, nrows=nrows)
    print(f"Train loaded: {train.shape}")

    if os.path.exists(feat_path):
        feats = pd.read_csv(feat_path, nrows=nrows)
        print(f"Features loaded: {feats.shape}")
    else:
        feats = None
        print("Feature file not found — skipping feature join.")

    return train, feats


def eda_aliccp(train, feats=None):
    print("\n" + "="*60)
    print("ALI-CCP — Exploratory Data Analysis")
    print("="*60)

    # ── Column detection ───────────────────────────────────────
    # Ali-CCP columns vary slightly by release; auto-detect click/conversion
    click_col = [c for c in train.columns if "click" in c.lower()]
    conv_col  = [c for c in train.columns if "buy" in c.lower()
                 or "conver" in c.lower() or "purchase" in c.lower()]
    user_col  = [c for c in train.columns if "user" in c.lower()]
    item_col  = [c for c in train.columns if "item" in c.lower()
                 or "adgroup" in c.lower()]

    print("\n[1] Columns:", list(train.columns))
    print(f"    Detected — click: {click_col}, conversion: {conv_col}")
    print(f"    user cols: {user_col}, item cols: {item_col}")
    print("\n[2] Missing values (%):\n",
          (train.isnull().mean()*100).round(2).sort_values(ascending=False))

    # ── 2. CTR and CVR ────────────────────────────────────────
    if click_col:
        ctr = train[click_col[0]].mean()
        print(f"\n[3] Click-Through Rate (CTR): {ctr:.4%}")
    if conv_col:
        # CVR = conversions / clicks (post-click only)
        clicked = train[train[click_col[0]] == 1] if click_col else train
        cvr = clicked[conv_col[0]].mean()
        print(f"    Conversion Rate (CVR, post-click): {cvr:.4%}")
        print(f"    Total conversions: {clicked[conv_col[0]].sum():,}")
        print(f"    Total clicks:      {len(clicked):,}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Label distribution
    if conv_col and click_col:
        labels = ["Impression\n(no click)", "Click\n(no convert)", "Conversion"]
        n_imp   = (train[click_col[0]] == 0).sum()
        n_click = ((train[click_col[0]] == 1) & (train[conv_col[0]] == 0)).sum()
        n_conv  = (train[conv_col[0]] == 1).sum()
        axes[0].bar(labels, [n_imp, n_click, n_conv],
                    color=["#BDBDBD", "#90CAF9", "#A5D6A7"])
        axes[0].set_title("Sample Distribution")
        axes[0].set_ylabel("Count")
        axes[0].set_yscale("log")

    # ── 3. User sparsity ──────────────────────────────────────
    if user_col:
        user_activity = train.groupby(user_col[0]).size()
        axes[1].hist(user_activity.clip(upper=user_activity.quantile(0.99)),
                     bins=40, color="#CE93D8", edgecolor="white")
        axes[1].set_title("Interactions per User")
        axes[1].set_xlabel("Number of interactions")
        axes[1].set_ylabel("User count")
        print(f"\n[4] User sparsity:")
        print(f"    Unique users : {user_activity.shape[0]:,}")
        print(f"    Median interactions/user: {user_activity.median():.0f}")
        print(f"    Mean   interactions/user: {user_activity.mean():.1f}")
        print(f"    Users with ≤3 interactions: "
              f"{(user_activity <= 3).mean():.1%}")

    # ── 4. Item sparsity ──────────────────────────────────────
    if item_col:
        item_activity = train.groupby(item_col[0]).size()
        axes[2].hist(item_activity.clip(upper=item_activity.quantile(0.99)),
                     bins=40, color="#FFCC80", edgecolor="white")
        axes[2].set_title("Interactions per Item")
        axes[2].set_xlabel("Number of interactions")
        axes[2].set_ylabel("Item count")
        print(f"\n[5] Item sparsity:")
        print(f"    Unique items : {item_activity.shape[0]:,}")
        print(f"    Median interactions/item: {item_activity.median():.0f}")
        print(f"    Items with ≤3 interactions: "
              f"{(item_activity <= 3).mean():.1%}")

    plt.suptitle("Ali-CCP Dataset — Key Statistics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("aliccp_eda.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── 5. Sparsity matrix ────────────────────────────────────
    if user_col and item_col:
        n_users = train[user_col[0]].nunique()
        n_items = train[item_col[0]].nunique()
        n_inter = len(train)
        density = n_inter / (n_users * n_items)
        print(f"\n[6] Interaction matrix density:")
        print(f"    Users × Items = {n_users:,} × {n_items:,}")
        print(f"    Interactions  = {n_inter:,}")
        print(f"    Density       = {density:.6%}  (very sparse ✓)")

    # ── 6. Feature type overview ──────────────────────────────
    if feats is not None:
        print("\n[7] Feature store overview:")
        print(f"    Shape: {feats.shape}")
        print(f"    Columns: {list(feats.columns)}")
        print(f"    Sample:\n{feats.head(3)}")

    print("\n✅ Ali-CCP EDA complete. Charts saved.")


# ============================================================
#  SECTION 3 — SIDE-BY-SIDE SUMMARY
# ============================================================

def print_summary():
    summary = {
        "": ["Ali-CCP", "Criteo"],
        "Total samples": ["~1 billion impressions", "~16 million clicks"],
        "Positive rate (CVR)": ["~2.1%", "~20%"],
        "Time span": ["8 days", "90 days"],
        "Max conversion delay": ["Not explicit", "30 days"],
        "User features": ["Age, gender, occupation ✓", "Anonymous hash"],
        "Item features": ["Category, brand, price ✓", "Anonymised"],
        "Text/LLM usable": ["✅ Yes", "❌ No"],
        "User sequence": ["✅ Yes", "❌ No"],
        "DF research standard": ["✅ Yes", "✅ Yes"],
        "Dissertation primary": ["✅ PRIMARY", "Supplementary (DF only)"],
    }
    df = pd.DataFrame(summary).set_index("")
    print("\n" + "="*60)
    print("DATASET COMPARISON SUMMARY")
    print("="*60)
    print(df.to_string())


# ============================================================
#  MAIN — run whichever section applies
# ============================================================

if __name__ == "__main__":
    print_summary()

    # ── Criteo ────────────────────────────────────────────────
    # Uncomment after downloading from:
    # http://go.criteo.net/criteo-research-search-conversion.tar.gz
    #
    # criteo = load_criteo(CRITEO_PATH, nrows=500_000)
    # eda_criteo(criteo)

    # ── Ali-CCP ───────────────────────────────────────────────
    # Uncomment after downloading from Tianchi:
    # https://tianchi.aliyun.com/dataset/408
    #
    # train, feats = load_aliccp(ALICCP_TRAIN, ALICCP_FEATS, nrows=500_000)
    # eda_aliccp(train, feats)

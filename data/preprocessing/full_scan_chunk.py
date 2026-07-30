"""
Resumable full-file exact scan for sample_skeleton_train.csv.
Run repeatedly (each call processes a time-boxed chunk, saves state, resumes
from the saved byte offset next time) until it reports DONE.

Tracks exact n_total / n_click / n_purchase (no sampling error), plus a
reservoir sample (Algorithm R) of full parsed rows (incl. item_id) for
distributional stats (item sparsity, feature_num, common_feature_index reuse).
"""
import json, os, pickle, random, time

WORK_DIR = r"E:\BaiduNetdiskDownload\Dataset\_processed"
PATH = r"E:\BaiduNetdiskDownload\Dataset\sample_train\sample_skeleton_train.csv"
STATE_PATH = os.path.join(WORK_DIR, "scan_state.json")
RESERVOIR_PATH = os.path.join(WORK_DIR, "scan_reservoir.pkl")
RESERVOIR_SIZE = 50_000
TIME_BUDGET_SEC = 30

TRIPLE_SEP, FIELD_SEP, VALUE_SEP = "\x01", "\x02", "\x03"

def parse_item_id(blob):
    for tok in blob.split(TRIPLE_SEP):
        if not tok:
            continue
        try:
            fid, rest = tok.split(FIELD_SEP, 1)
        except ValueError:
            continue
        if fid == "205":
            try:
                feat_id, _ = rest.split(VALUE_SEP, 1)
            except ValueError:
                continue
            return feat_id
    return None

def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            state = json.load(f)
        with open(RESERVOIR_PATH, "rb") as f:
            reservoir = pickle.load(f)
        return state, reservoir
    state = {"offset": 0, "n_total": 0, "n_click": 0, "n_purchase": 0, "done": False}
    reservoir = []
    return state, reservoir

def save_state(state, reservoir):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)
    with open(RESERVOIR_PATH, "wb") as f:
        pickle.dump(reservoir, f)

def main():
    state, reservoir = load_state()
    if state["done"]:
        print("Already DONE. n_total =", state["n_total"])
        return

    file_size = os.path.getsize(PATH)
    t0 = time.time()
    n_total = state["n_total"]
    n_click = state["n_click"]
    n_purchase = state["n_purchase"]

    with open(PATH, "r", encoding="utf-8", errors="replace") as f:
        f.seek(state["offset"])
        while True:
            if time.time() - t0 > TIME_BUDGET_SEC:
                break
            line = f.readline()
            if not line:
                state["done"] = True
                break
            parts = line.rstrip("\n").split(",", 5)
            if len(parts) < 6:
                continue
            _sid, click, purchase, cidx, fnum, blob = parts
            is_click = (click == "1")
            is_purchase = (purchase == "1")
            n_click += is_click
            n_purchase += is_purchase

            # reservoir sampling (Algorithm R) — only parse blob for rows that
            # actually get inserted, keeping the hot path cheap
            if len(reservoir) < RESERVOIR_SIZE:
                reservoir.append({
                    "click": int(is_click), "purchase": int(is_purchase),
                    "common_feature_index": cidx, "feature_num": int(fnum),
                    "item_id": parse_item_id(blob),
                })
            else:
                j = random.randint(0, n_total)
                if j < RESERVOIR_SIZE:
                    reservoir[j] = {
                        "click": int(is_click), "purchase": int(is_purchase),
                        "common_feature_index": cidx, "feature_num": int(fnum),
                        "item_id": parse_item_id(blob),
                    }
            n_total += 1

        state["offset"] = f.tell()

    state["n_total"] = n_total
    state["n_click"] = n_click
    state["n_purchase"] = n_purchase
    save_state(state, reservoir)

    elapsed = time.time() - t0
    pct = 100 * state["offset"] / file_size
    print(f"Chunk done in {elapsed:.1f}s. Progress: {pct:.1f}%  "
          f"({state['offset']:,} / {file_size:,} bytes)")
    print(f"Cumulative: n_total={n_total:,}  n_click={n_click:,}  n_purchase={n_purchase:,}")
    if n_click:
        print(f"Running CTR={n_click/n_total:.4%}  CVR(post-click)={n_purchase/n_click:.4%}  "
              f"CVR(overall)={n_purchase/n_total:.5%}")
    if state["done"]:
        print("\n*** FULL SCAN COMPLETE ***")

if __name__ == "__main__":
    main()

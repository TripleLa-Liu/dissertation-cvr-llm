"""
Quick diagnostic: extended session-side k-core table, computed directly from
the already-saved aliccp_degree_counters.pkl — no rescan of the 10GB skeleton
file needed, runs in under a second.

Why this is needed: the first filter_and_join.py run (K_ITEM=50, K_SESSION=100)
kept 9,964,603 rows — still ~2-5x above the ~2-5M target — while item/session
entity counts (140,782 / 94,964) already landed inside the 50K-200K target.
Since raising K_ITEM further would mostly just shrink the item vocabulary
(which is already fine), the next move is to push K_SESSION higher — session
degree is the main volume lever (see degree_distribution_scan.py docstring).
This script extends that table past k=100 so you can pick a K_SESSION that
gets rows into range without dropping session count too far below 50K.
"""
import pickle

with open("aliccp_degree_counters.pkl", "rb") as f:
    d = pickle.load(f)
session_counter = d["session_counter"]

print(f"{'min interactions':>18} | {'sessions kept':>14} | {'session-rows kept':>18}")
print("-" * 60)
for k in [100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000]:
    kept_sessions = sum(1 for c in session_counter.values() if c >= k)
    kept_rows = sum(c for c in session_counter.values() if c >= k)
    print(f"{k:>18,} | {kept_sessions:>14,} | {kept_rows:>18,}")

print("\nRemember: this is the MARGINAL session-only count (item filter not")
print("applied). The actual combined row count after also requiring item")
print("degree >= K_ITEM will be somewhat lower than shown here — use this")
print("table to narrow down a K_SESSION to try, then rerun filter_and_join.py")
print("(step 1 alone takes ~106s locally, so trying 2-3 values is cheap).")

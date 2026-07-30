"""
Extended session-side k-core threshold table, computed directly from the
saved aliccp_degree_counters.pkl (no rescan of the 10GB skeleton file needed).

Session degree is the main lever for total row-count control (see
degree_distribution_scan.py): item thresholds mostly affect vocabulary size,
while session thresholds control row volume. This extends the marginal
threshold table past k=100 to help pick a K_SESSION value.
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

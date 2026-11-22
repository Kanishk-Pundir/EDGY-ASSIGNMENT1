import csv, sys
import numpy as np

path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
print(f"file: {path}   requests: {len(rows)}")
print(f"first request: {float(rows[0]['round_trip']):.1f} ms round trip")
print(f"{'stage':24s} {'p50 ms':>9} {'p95 ms':>9} {'p99 ms':>9}   (requests 2..{len(rows)})")
for k in rows[0]:
    if k == "i":
        continue
    v = np.array([float(r[k]) for r in rows[1:]])
    print(f"{k:24s} {np.percentile(v, 50):9.3f} {np.percentile(v, 95):9.3f} {np.percentile(v, 99):9.3f}")
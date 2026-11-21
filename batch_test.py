"""400 single /predict calls vs one /predict_batch call. Usage: python batch_test.py <url>"""
import sys
import time
import numpy as np
import requests

url = sys.argv[1].rstrip("/")
rng = np.random.default_rng(1)
rows = [{"battery_v": round(float(rng.uniform(3.4, 4.2)), 3),
         "temp_c": round(float(rng.uniform(10, 45)), 1),
         "rings_per_day": int(rng.integers(0, 30)),
         "wifi_rssi_dbm": int(rng.integers(-88, -45)),
         "age_months": int(rng.integers(0, 37))} for _ in range(400)]

s = requests.Session()
s.get(url + "/health", timeout=30)             # warm up the connection first

t0 = time.perf_counter()
for r in rows:
    s.post(url + "/predict", json=r, timeout=30)
t_single = time.perf_counter() - t0

times = []
for _ in range(5):                             # repeat: the first call may include warm-up
    t0 = time.perf_counter()
    resp = s.post(url + "/predict_batch", json={"rows": rows}, timeout=60)
    times.append(time.perf_counter() - t0)
    print("  batch call:", f"{times[-1]:.3f} s", resp.json()["timings_ms"])
t_batch = sorted(times)[len(times) // 2]       # median

print(f"400 single calls: {t_single:.2f} s    one batch call (median of 5): {t_batch:.3f} s    ratio: {t_single / t_batch:.0f}x")
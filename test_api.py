"""
test_api.py -- self-check for the doorbell API, mirroring the instructor's six checks.

    python test_api.py                                        # http://localhost:8000
    python test_api.py https://<name>-8000.app.github.dev     # the public endpoint

Prints PASS/FAIL per check and exits with code 1 if anything fails.
"""
import json
import math
import sys

import requests

url = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
ok = {"battery_v": 3.8, "temp_c": 28, "rings_per_day": 8, "wifi_rssi_dbm": -66, "age_months": 18}
# json.dumps (not requests' json=) so NaN and Infinity can be sent the way a broken sensor would.
post = lambda path, body: requests.post(url + path, data=json.dumps(body), timeout=30,
                                        headers={"content-type": "application/json"})
failures = 0


def check(name, passed, extra=""):
    global failures
    failures += not passed
    print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f"   ({extra})" if extra else ""))


# 1. Health ---------------------------------------------------------------------------------
health = requests.get(url + "/health", timeout=30).json()
check("health says ok, team 5", health.get("status") == "ok" and health.get("team") == 5, health)

# 2. A good reading, and stage timing on it --------------------------------------------------
r = post("/predict", ok)
body = r.json()
check("good reading answered", r.status_code == 200 and isinstance(body.get("days_left"), float), body)
need = {"parse", "normalise", "inference", "build", "server_total"}
check("all five timing keys present", need <= set(body.get("timings_ms", {})))

# 3. Per-layer table ------------------------------------------------------------------------
info = requests.get(url + "/model_info", timeout=30).json()
layers = info["layers"]
chain = layers[0]["in"] == 5 and layers[-1]["out"] == 1 and all(
    a["out"] == b["in"] for a, b in zip(layers, layers[1:]))
check("layers connect 5 -> ... -> 1", chain, " -> ".join([str(layers[0]["in"])] + [str(l["out"]) for l in layers]))
rows_ok = all(l["params"] == l["in"] * l["out"] + l["out"] and l["bytes_f32"] == 4 * l["params"]
              and l["macs"] == l["in"] * l["out"] for l in layers)
check("every layer's params, bytes and MACs", rows_ok)
tot = sum(l["params"] for l in layers)
check("totals add up", info["total_params"] == tot and info["total_bytes_f32"] == 4 * tot
      and info["total_bytes_int8"] == tot and info["total_macs"] == sum(l["macs"] for l in layers))
check("fits flags right (300 KB = 307,200 B)",
      info["fits_300kb_sram_f32"] == (4 * tot <= 307200) and info["fits_300kb_sram_int8"] == (tot <= 307200))
check("/model_info total_params == /health params", info["total_params"] == health["params"])

# 4. Broken readings: every one must be refused with a 4xx, never answered -------------------
bad = {
    "0 V": {**ok, "battery_v": 0},
    "-127 C": {**ok, "temp_c": -127},
    "0 dBm": {**ok, "wifi_rssi_dbm": 0},
    "negative rings": {**ok, "rings_per_day": -5},
    "negative age": {**ok, "age_months": -1},
    "text value": {**ok, "temp_c": "abc"},
    "null value": {**ok, "battery_v": None},
    "NaN": {**ok, "battery_v": math.nan},
    "Infinity": {**ok, "temp_c": math.inf},
    "-Infinity": {**ok, "wifi_rssi_dbm": -math.inf},
    "missing field": {k: v for k, v in ok.items() if k != "battery_v"},
    "empty object": {},
}
for name, b in bad.items():
    code = post("/predict", b).status_code
    check(f"refused: {name}", 400 <= code < 500, code)
code = requests.post(url + "/predict", data="not json", headers={"content-type": "application/json"},
                     timeout=30).status_code
check("refused: invalid JSON", 400 <= code < 500, code)
code = post("/predict_batch", {"rows": [ok, {**ok, "battery_v": math.nan}]}).status_code
check("refused: batch with one bad row", 400 <= code < 500, code)

# Real readings just outside the training data must still be answered.
for name, b in {"3.3 V, 5 C": {**ok, "battery_v": 3.3, "temp_c": 5},
                "-95 dBm, 48 months": {**ok, "wifi_rssi_dbm": -95, "age_months": 48}}.items():
    code = post("/predict", b).status_code
    check(f"answered: {name} (realistic, outside the data)", code == 200, code)

# 5. Batch gives the same answers as single calls --------------------------------------------
rows = [{**ok, "battery_v": round(3.4 + 0.016 * i, 3), "age_months": i % 37} for i in range(50)]
single = [post("/predict", x).json()["days_left"] for x in rows]
batch = post("/predict_batch", {"rows": rows}).json()["days_left"]
diff = max(abs(a - b) for a, b in zip(single, batch))
check("batch matches 50 single calls", len(batch) == 50 and diff < 0.011, f"max difference {diff:.3f} days")

print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
sys.exit(1 if failures else 0)

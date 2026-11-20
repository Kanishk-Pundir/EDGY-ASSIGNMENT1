"""Self-check for the doorbell API. Usage: python test_api.py [url]   (default http://localhost:8000)"""
import sys
import requests

url = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
ok = {"battery_v": 3.8, "temp_c": 28, "rings_per_day": 8, "wifi_rssi_dbm": -66, "age_months": 18}
post = lambda path, body: requests.post(url + path, json=body, timeout=30)

print("health:", requests.get(url + "/health", timeout=30).json())

r = post("/predict", ok)
print("good reading:", r.status_code, r.json())
need = {"parse", "normalise", "inference", "build", "server_total"}
print("all timing keys present:", need <= set(r.json().get("timings_ms", {})))

bad = {
    "0 V": {**ok, "battery_v": 0},
    "-127 C": {**ok, "temp_c": -127},
    "0 dBm": {**ok, "wifi_rssi_dbm": 0},
    "negative rings": {**ok, "rings_per_day": -5},
    "text value": {**ok, "temp_c": "abc"},
    "missing field": {k: v for k, v in ok.items() if k != "battery_v"},
}
for name, body in bad.items():
    code = post("/predict", body).status_code
    print(f"bad ({name}): {code}", "OK" if code == 422 else "<-- should be 422")

code = post("/predict", {**ok, "battery_v": 3.3, "temp_c": 5}).status_code
print("3.3 V, 5 C (realistic, just outside data):", code, "OK" if code == 200 else "<-- should be 200")

rows = [{**ok, "battery_v": 3.4 + 0.016 * i} for i in range(50)]
single = [post("/predict", x).json()["days_left"] for x in rows]
batch = post("/predict_batch", {"rows": rows}).json()["days_left"]
print("batch matches 50 singles:", max(abs(a - b) for a, b in zip(single, batch)) < 0.011)

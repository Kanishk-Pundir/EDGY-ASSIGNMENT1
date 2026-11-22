"""
app/main.py -- Team 5 (Edgy) doorbell battery API.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

GET  /health          -> {"status": "ok", "team": 5, "params": 1281, "val_mae": 4.88}
GET  /model_info      -> the per-layer table: params, bytes, MACs, and whether it fits the doorbell
POST /predict         one reading  -> {"days_left": 52.1, "timings_ms": {...}}
POST /predict_batch   {"rows": [reading, ...]} -> {"days_left": [...], "timings_ms": {...}}
A reading that is out of range, missing, non-numeric, NaN or infinite gets HTTP 422 with
the reason, never a guess.
"""
import json
import time

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.model import FEATURES, BatteryNet
from app.stats import layer_table

cfg = json.load(open("model/config.json"))
net = BatteryNet(tuple(cfg["hidden"]))
net.load_state_dict(torch.load("model/model.pt", weights_only=True))
net.eval()
torch.set_num_threads(1)
MU, SD = np.array(cfg["mu"], np.float32), np.array(cfg["sd"], np.float32)
N_PARAMS = sum(p.numel() for p in net.parameters())
MODEL_INFO = layer_table(net)          # the network never changes while serving: count it once
assert MODEL_INFO["total_params"] == N_PARAMS, "layer_table disagrees with PyTorch's count"

# Warm-up: PyTorch's first forward pass does one-time set-up (Mission 3 measured hundreds of ms).
# Pay it here at start-up, for one row and for a batch, so no real request pays it.
with torch.no_grad():
    for n in (1, 64):
        net(torch.zeros(n, len(FEATURES)))


class Reading(BaseModel):
    # Mission 3: limits come from what a real sensor can report, NOT from the training data's
    # min/max, so a real reading just outside the data is still answered while glitches
    # (0 V, -127 C, 0 dBm) are refused with HTTP 422.
    # allow_inf_nan=False: a dead sensor can send NaN or Infinity. Pydantic accepts those in a
    # float field by default, and NaN slips past ge/le checks, so refuse them explicitly.
    model_config = ConfigDict(allow_inf_nan=False)

    battery_v: float = Field(ge=2.5, le=4.5)        # Li-ion cell, ~3.0-4.2 V in normal use
    temp_c: float = Field(ge=-40, le=85)            # typical electronics operating range
    rings_per_day: float = Field(ge=0, le=200)      # a count: cannot be negative
    wifi_rssi_dbm: float = Field(ge=-100, le=-10)   # always negative; 0 dBm is impossible
    age_months: float = Field(ge=0, le=120)         # cannot be negative


class Batch(BaseModel):
    rows: list[Reading] = Field(min_length=1, max_length=10000)


app = FastAPI(title="Doorbell battery API")
ms = lambda t0: round((time.perf_counter() - t0) * 1000, 3)


@app.get("/model_info")
def model_info():
    # Missions 2 + 4: computed from the network's own layers by app/stats.py (not typed in),
    # so it stays right if the layer sizes change. mission4.py uses the same function.
    return MODEL_INFO


@app.get("/health")
def health():
    return {"status": "ok", "team": cfg["team"], "params": N_PARAMS, "val_mae": cfg["val_mae"]}


async def run(request: Request, schema):
    t_all = time.perf_counter(); t = {}
    t0 = time.perf_counter()
    try:
        data = schema.model_validate_json(await request.body())
    except ValidationError as e:
        # include_input=False: the rejected value may be NaN/Infinity, which JSON cannot carry,
        # so echoing it back would crash the error reply itself (HTTP 500).
        detail = json.loads(e.json(include_input=False, include_url=False))
        return JSONResponse(status_code=422, content={"error": "bad reading", "detail": detail})
    t["parse"] = ms(t0)
    rows = data.rows if isinstance(data, Batch) else [data]

    t0 = time.perf_counter()
    x = (np.array([[getattr(r, f) for f in FEATURES] for r in rows], np.float32) - MU) / SD
    t["normalise"] = ms(t0)

    t0 = time.perf_counter()
    with torch.no_grad():
        y = net(torch.from_numpy(x)).numpy()
    t["inference"] = ms(t0)

    t0 = time.perf_counter()
    days = [round(float(v), 2) for v in np.maximum(y, 0)]
    body = {"days_left": days if isinstance(data, Batch) else days[0]}
    t["build"] = ms(t0)

    t["server_total"] = ms(t_all)
    body["timings_ms"] = t
    return body


# The endpoints read the raw body (so parse can be timed), which hides the body from /docs.
# openapi_extra puts the schema back, so readings can be tried by hand at /docs.
def _body(schema):
    return {"requestBody": {"required": True, "content": {"application/json": {"schema": schema}}}}


READING_SCHEMA = Reading.model_json_schema()
BATCH_SCHEMA = {"type": "object", "required": ["rows"],
                "properties": {"rows": {"type": "array", "items": READING_SCHEMA}}}


@app.post("/predict", openapi_extra=_body(READING_SCHEMA))
async def predict(request: Request):
    return await run(request, Reading)


@app.post("/predict_batch", openapi_extra=_body(BATCH_SCHEMA))
async def predict_batch(request: Request):
    return await run(request, Batch)

"""
app/main.py -- STARTER battery API. Look for TODO.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

GET  /health          -> {"status": "ok", "team": 7, "params": 1281, "val_mae": 3.4}
GET  /model_info      -> the per-layer table: params, bytes, MACs, and whether it fits the doorbell
POST /predict         one reading  -> {"days_left": 52.1, "timings_ms": {...}}
POST /predict_batch   {"rows": [reading, ...]} -> {"days_left": [...], "timings_ms": {...}}
A reading that is out of range gets HTTP 422 with the reason, never a guess.
"""
import json
import time

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from app.model import FEATURES, BatteryNet

cfg = json.load(open("model/config.json"))
net = BatteryNet(tuple(cfg["hidden"]))
net.load_state_dict(torch.load("model/model.pt", weights_only=True))
net.eval()
torch.set_num_threads(1)
MU, SD = np.array(cfg["mu"], np.float32), np.array(cfg["sd"], np.float32)
N_PARAMS = sum(p.numel() for p in net.parameters())

class Reading(BaseModel):
    # Mission 3: limits come from what a real sensor can report, NOT from the training data's
    # min/max, so a real reading just outside the data is still answered while glitches
    # (0 V, -127 C, 0 dBm) are refused with HTTP 422.
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
    # TODO Mission 2 + 4: return your per-layer table, worked out FROM THE NETWORK (net.net),
    # not typed in by hand, so it stays right if you change the layer sizes. Exactly these keys:
    # {"layers": [{"name": "linear1", "in": 5, "out": 32, "params": ..., "bytes_f32": ..., "macs": ...}, ...],
    #  "total_params": ..., "total_bytes_f32": ..., "total_bytes_int8": ..., "total_macs": ...,
    #  "fits_300kb_sram_f32": true/false, "fits_300kb_sram_int8": true/false}
    # Count only the Linear layers (ReLU has no parameters). Clue: a Linear layer m has
    # m.in_features and m.out_features.
    layers = []

    total_params = 0
    total_bytes_f32 = 0
    total_macs = 0

    for layer in net.net:
        if isinstance(layer, torch.nn.Linear):
            params = (layer.in_features * layer.out_features + layer.out_features)

            bytes_f32 = params * 4

            macs = layer.in_features * layer.out_features

            total_params += params
            total_bytes_f32 += bytes_f32
            total_macs += macs

            layers.append({
                "name": f"linear{len(layers) + 1}",
                "in": layer.in_features,
                "out": layer.out_features,
                "params": params,
                "bytes_f32": bytes_f32,
                "macs": macs
            })

    total_bytes_int8 = total_params

    fits_f32 = total_bytes_f32 <= 300 * 1024
    fits_int8 = total_bytes_int8 <= 300 * 1024

    return {
    "layers": layers,
    "total_params": total_params,
    "total_bytes_f32": total_bytes_f32,
    "total_bytes_int8": total_bytes_int8,
    "total_macs": total_macs,
    "fits_300kb_sram_f32": fits_f32,
    "fits_300kb_sram_int8": fits_int8
    }


@app.get("/health")
def health():
    return {"status": "ok", "team": cfg["team"], "params": N_PARAMS, "val_mae": cfg["val_mae"]}


async def run(request: Request, schema):
    t_all = time.perf_counter(); t = {}
    t0 = time.perf_counter()
    try:
        data = schema.model_validate_json(await request.body())
    except ValidationError as e:
        return JSONResponse(status_code=422, content={"error": "bad reading", "detail": json.loads(e.json())})
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


@app.post("/predict")
async def predict(request: Request):
    return await run(request, Reading)


@app.post("/predict_batch")
async def predict_batch(request: Request):
    return await run(request, Batch)

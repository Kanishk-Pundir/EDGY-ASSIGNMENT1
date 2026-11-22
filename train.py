"""
train.py -- Clean the data, fit baselines, train the network, save it for the server.

    python train.py --team 5 --hidden 32,32
Writes model/model.pt and model/config.json, which app/main.py loads.

Done (Missions 1 & 2): Data cleaning, baselines, input normalisation, and network training.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from app.model import FEATURES, BatteryNet

ap = argparse.ArgumentParser()
ap.add_argument("--team", type=int, required=True)
ap.add_argument("--hidden", default="32,32")
ap.add_argument("--epochs", type=int, default=1500)
a = ap.parse_args()
hidden = tuple(int(h) for h in a.hidden.split(","))

df = pd.read_csv(f"data/team_{a.team:02d}_train.csv")
# Done (Mission 1): Drop 24 sensor glitch rows across 3 failure modes:
# - battery_v == 0 (7 rows): open circuit or disconnected ADC channel.
# - temp_c == -127 (11 rows): DS18B20 digital 1-Wire bus fault / reset sentinel.
# - wifi_rssi_dbm == 0 (6 rows): impossible 1 mW RF power; uninitialized driver state.
# Natural zeros (rings_per_day == 0 and age_months == 0) are valid and preserved.
bad = (
    (df["battery_v"] == 0)|
    (df["temp_c"] == -127)|
    (df["wifi_rssi_dbm"] == 0)
)
print(f"{len(df)} rows, {bad.sum()} glitch rows dropped")
df = df[~bad]

rng = np.random.default_rng(0)
idx = rng.permutation(len(df)); n_val = len(df) // 5
val, tr = df.iloc[idx[:n_val]], df.iloc[idx[n_val:]]
X, y = tr[FEATURES].to_numpy(np.float32), tr.days_left.to_numpy(np.float32)
Xv, yv = val[FEATURES].to_numpy(np.float32), val.days_left.to_numpy(np.float32)
# Done (Mission 2): Input normalisation. mu and sd are computed strictly from the TRAINING
# rows to prevent data leakage from the validation set into the model.
mu = np.mean(X, axis=0)
sd = np.std(X, axis=0)



mae = lambda p, t: float(np.mean(np.abs(p - t)))
print(f"baseline (predict the mean): {mae(np.full_like(yv, y.mean()), yv):.2f} days")
# Done (Mission 1): Linear regression baseline using ordinary least squares (np.linalg.lstsq)
# on [X, 1] to quantify linear predictive signal in the sensor features.

train_mat = np.ones((X.shape[0], 1))
train_mat = np.column_stack((X, train_mat))

train_mat_val = np.ones((Xv.shape[0], 1))
train_mat_val = np.column_stack((Xv, train_mat_val))

coefficients = np.linalg.lstsq(train_mat, y, rcond=None)[0]

predictions = train_mat_val @ coefficients

print(f"baseline (linear regression): {mae(predictions, yv):.2f} days")

## The mean prediction baseline achieved a validation MAE of 57.27 days. Linear regression using NumPy's least-squares method achieved 22.84 days, reducing the validation error by approximately 60.1%. This indicates that the input features contain useful predictive information.

torch.manual_seed(0)
net = BatteryNet(hidden)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(y)
Xvt = torch.tensor((Xv - mu) / sd)
for ep in range(a.epochs):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 128):
        b = perm[i:i + 128]
        loss = torch.nn.functional.mse_loss(net(Xt[b]), yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 300 == 0 or ep == a.epochs - 1:
        with torch.no_grad():
            tr_m = mae(net(Xt).numpy(), y)
            val_m = mae(net(Xvt).numpy(), yv)
            print(f"  epoch {ep:4d}  train MAE {tr_m:.2f} days  val MAE {val_m:.2f} days")
with torch.no_grad():
    train_mae = mae(net(Xt).numpy(), y)
    val_mae = mae(net(Xvt).numpy(), yv)
print(f"network {hidden}: train MAE {train_mae:.2f} days, val MAE {val_mae:.2f} days")

os.makedirs("model", exist_ok=True)
torch.save(net.state_dict(), "model/model.pt")
json.dump(dict(team=a.team, hidden=list(hidden), mu=mu.tolist(), sd=sd.tolist(),
               features=FEATURES, val_mae=round(val_mae, 2)), open("model/config.json", "w"), indent=2)
print("saved model/model.pt and model/config.json")
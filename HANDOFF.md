# Handoff: Team 5 (Edgy), Edge AI Assignment 1, for Mission 4

Deadline: **Monday, 23 November 2026**. Read `README.md` first (the assignment brief), then this file.

## 1. The project in one paragraph

Train a small neural network that predicts `days_left` of a doorbell battery from five sensor readings, and serve it as a web API (FastAPI) from a GitHub Codespace. An instructor script checks the endpoint: health, accuracy on 400 hidden readings, the per-layer table at `/model_info`, five broken readings (must be refused), stage timing on every reply, and batch against single answers. We hand in two things: the **endpoint URL** and the **repo link**.

## 2. Where we are

| Mission | Status | Notes |
|---|---|---|
| 1. Data and baselines | Done (`train.py`) | Glitch rule drops 24 rows (battery 0 V, temp -127 C, RSSI 0 dBm). Baselines on our split: mean about 57 days MAE, linear about 23 days MAE. |
| 2. Normalisation, model, `/model_info` | Done | Normalisation uses training rows only. `/model_info` is computed from the network, not typed in. Model-size sweep and plots are documented in `docs/mission1_2_report.md` and `plots/`. |
| 3. Ranges, timing, benchmarks | Done | See `docs/mission3_report.md` and `results/`. |
| 4. Would it fit in the doorbell? | Done | See docs/mission4_report.md and mission4.py. |

**Current model:** hidden layers 32,32 (5 → 32 → 32 → 1), **1,281 params**, 1,216 MACs, 5,124 bytes as float32, 1,281 bytes as int8, validation MAE 4.90 days. Normalisation values are stored in `model/config.json`.

## 3. Repo layout

```
README.md              assignment brief (read first)
train.py               training script; writes model/model.pt and model/config.json
app/model.py           BatteryNet definition (shared by training and server)
app/main.py            the API: ranges, timing, /model_info, /predict, /predict_batch
client/bench.py        latency benchmark (run, do not edit)
data/team_05_train.csv our data (use only this file)
model/                 trained model (must be committed: the server loads it)
test_api.py            our correctness check (ranges, timing keys, batch equals singles)
batch_test.py          400 single calls against one batch call
handshake_test.py      splits connection set-up into DNS, TCP, TLS
docs/                  Mission reports (Missions 1-2, 3, 4)
plots/                 generated plots (Missions 1-2, 4)
results/               benchmark CSV files (evidence)
```

## 4. Setup on your own machine

You need Python 3.11 and Git.

```
git clone <REPO URL>
cd <repo folder>
python -m venv venv
venv\Scripts\activate            # Windows PowerShell
# source venv/bin/activate       # Mac / Linux
pip install -r requirements.txt
```

Run the server and check it:

```
uvicorn app.main:app --port 8000
```

In a second terminal (same folder, venv active):

```
python test_api.py
```

Everything should print OK or True, ending with `batch matches 50 singles: True`. Run commands from the **top folder**, not from inside `app/`.

`train.py` is only needed if `model/` is missing or you change the model size:

```
python train.py --team 5 --hidden 32,32
```

## 5. The online endpoint (Codespace)

The checker talks to the Codespace, so the server must run there.

1. Open the repo's Codespace (GitHub, Code, Codespaces). It installs the requirements itself.
2. Start the server in a terminal and **leave that terminal running**:
   ```
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
3. In the **Ports** tab, set port 8000 to **Public**. It resets to Private whenever the Codespace restarts.
4. Test from your laptop: open `<endpoint URL>/health` in a browser, then run `python test_api.py <endpoint URL>`.
5. GitHub, Settings, Codespaces, Default idle timeout: set it to the maximum, and keep the Codespace tab open during the check window.

Endpoint URL: `https://animated-guacamole-v6rpvvg5v9943w4wq-8000.app.github.dev` (stays the same for this Codespace; it changes if a new Codespace is created)
Repo link: `https://github.com/Kanishk-Pundir/EDGY-ASSIGNMENT1`

**After changing `app/main.py` or `model/`:** commit, then restart uvicorn (Ctrl+C, run it again). It does not reload by itself.

## 6. Your job: Mission 4 (from the README)

- [ ] Bytes for weights and biases as float32 and int8, and the largest activation memory needed at once, for our model.
- [ ] `/model_info` already returns `total_bytes_int8`, `fits_300kb_sram_f32` and `fits_300kb_sram_int8` (done in `app/main.py`). Check them against a hand calculation.
- [ ] Use the same function to fill the table for 5 → w → w → 1 at w = 32, 128, 256, 512, 2048: parameters, float32 size, int8 size, fits in 300 KB SRAM (float32 and int8), fits in 8 MB PSRAM, MACs.
- [ ] Find the exact width where the weights stop fitting in 300 KB, at float32 and at int8.
- [ ] Estimate one prediction's time on the board. State the speed you assume and where it came from.
- [ ] Argue: should the doorbell call our server or run the model itself? Cover latency, privacy, cost, power, and what happens when Wi-Fi is down. Our Mission 3 numbers help here (a laptop request took 41 to 87 ms, of which the model was about 0.3 ms).

Counting rule: a `Linear(a, b)` layer has a×b weights plus b biases, a×b MACs per prediction, 4 bytes per float32 and 1 byte per int8.

**Decide and write down what "300 KB" means.** `app/main.py` uses 300 × 1024 = 307,200 bytes. If you prefer 300,000, change it in `main.py` and say so in your write-up. It only matters at the boundary.

## 7. Open items and risks

1. **Model size justified in the repo.** The final model is 32,32 (5 → 32 → 32 → 1) with 1,281 parameters and validation MAE 4.90 days (historical sweep: 4.89 days). The size sweep and plots are documented in `docs/mission1_2_report.md` and `plots/`.
2. **Consistency check.** `/health`'s `params` must equal `/model_info`'s `total_params`. They match for 32,32 (1,281).
3. **Home-router DNS problems.** One laptop's router (Jio) sometimes refused DNS lookups, which crashed benchmark runs. Fixes: set DNS to 8.8.8.8 and 1.1.1.1, or use a phone hotspot.
4. **Do not edit the same file in two places** (laptop and Codespace). Pull before you edit, push after.
5. **Do not commit `venv/`.** Add `venv/` and `.venv/` to `.gitignore`.
6. `/docs` (Swagger) cannot send a body to `/predict` because the endpoint reads the raw request. Use `test_api.py` or `curl` instead. An empty-body 422 there is expected.

## 8. Useful commands

```
python train.py --team 5 --hidden 32,32 --epochs 1500
uvicorn app.main:app --host 0.0.0.0 --port 8000
python test_api.py [URL]
python client/bench.py --url <URL> --n 200 [--no-keepalive]
python batch_test.py <URL>
```

Git:

```
git pull
git add -A
git commit -m "describe the change"
git push
```

## 9. Rules from the README

Use an LLM to explain, to check your reasoning and to explain error messages. Your measurements are the judge: test any claim an LLM makes. Everyone must be able to explain every line that is submitted.

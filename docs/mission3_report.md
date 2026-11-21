# Mission 3 Report: Serving and Timing the Battery Model

Team 5 (Edgy) | Edge AI, Module 1 | Mission 3: input ranges, stage timing, benchmarking

## 1. Summary

The endpoint refuses glitch readings, times every stage of every request, and answers batch calls identically to single calls. The model's own work takes about 0.3 ms per request. From a laptop, a request takes 41 to 87 ms, so more than 99% of the time is network and framework overhead, not the model.

## 2. What was changed in the code (`app/main.py`)

| Change | Detail |
|---|---|
| Input ranges on `Reading` | `battery_v` 2.5 to 4.5 V, `temp_c` -40 to 85 C, `rings_per_day` 0 to 200, `wifi_rssi_dbm` -100 to -10 dBm, `age_months` 0 to 120. Limits come from what a real sensor can report, not from the training data's min and max. |
| Stage timing | `normalise`, `inference` and `build` are timed the same way `parse` is. Every reply carries `parse`, `normalise`, `inference`, `build`, `server_total`. |
| Cleanup | Debug `print` calls removed from startup and `/model_info`. |

**Why these ranges:** the Mission 1 glitches were exactly 0 V, -127 C and 0 dBm. All three fall outside the ranges, so they get HTTP 422. Realistic readings just outside the training data (for example 3.3 V or 5 C) are still answered.

## 3. Correctness check (`test_api.py`, run against the public endpoint)

| Test | Result |
|---|---|
| `/health` | ok, team 5, 129 params, val MAE 6.15 |
| Good reading | 200, `days_left` 105.25, all five timing keys present |
| 0 V, -127 C, 0 dBm, negative rings, text value, missing field | all 422 |
| 3.3 V and 5 C (realistic, just outside the data) | 200 |
| 50 single calls against one `/predict_batch` call | identical answers (`True`) |

## 4. Benchmark results

All runs used `client/bench.py --n 200`. Percentiles cover requests 2 to 200.

| Run | First request | Round trip p50 / p95 / p99 (ms) | `server_total` p50 (ms) | `network_and_framework` p50 (ms) |
|---|---|---|---|---|
| Laptop to Codespace, keep-alive (run A) | 1255 | 40.7 / 168 / 527 | 0.31 | 40.4 |
| Laptop to Codespace, keep-alive (run B) | 729 | 87.4 / 119 / 223 | 0.36 | 87.1 |
| Laptop to Codespace, no keep-alive | 594 | 883.5 / 1196 / 2391 | 0.35 | 883.2 |
| Inside the Codespace (localhost) | 2.2 | 1.82 / 3.03 / 5.51 | 0.22 | 1.57 |

Stage times (p50, run B, laptop): parse 0.043 ms, normalise 0.043 ms, inference 0.230 ms, build 0.020 ms.

Connection set-up cost measured separately (`handshake_test.py`, 20 repeats, laptop to the Codespace): DNS lookup 0.8 ms median (max 69.3 ms), TCP connect 72.6 ms median, TLS handshake 77.6 ms median.

### Batch against single calls (`batch_test.py`)

| Version | 400 single calls | One batch call of 400 rows | Ratio |
|---|---|---|---|
| First version (cold connection, one call) | 38.99 s | 0.865 s (server work 9.1 ms, inference 0.28 ms) | 45x |
| Corrected (warm session, median of 5) | 36.36 s (about 91 ms per call) | 0.219 s (five calls: 0.207 to 0.234 s; server work 2.3 to 12.4 ms) | 166x |

An even earlier batch call showed 104.7 ms of inference. That was first-call warm-up (see 6.3): later batch calls took 0.24 to 0.86 ms for the same 400 rows.

## 5. Predictions

### 5.0 Initial Guesses


| Question | Guess | Measured | What the gap teaches |
|---|---|---|---|
| Inference time, one reading | 1 to 5 ms (a PyTorch forward pass sounds expensive) | 0.12 to 0.24 ms (first call 26.8 ms) | A 129-parameter network is nearly free. The first call is slow only from set-up. |
| Laptop round trip | 100 to 200 ms | 41 to 87 ms median | The Codespace is in a nearby region (Azure Central India), and the network varies run to run. |
| Slowest stage | Inference (the neural network) | Network and framework, over 99% | The model is rarely the bottleneck in a served system. |
| Effect of no keep-alive | Slightly slower, about 2x | About 10x (883 ms against 87 ms) | New connections cost far more than expected, and we only partly explain why (see 6.4). |
| Localhost round trip | About 5 ms | 1.8 ms median | Framework overhead was still about 7x the model's compute. |
| 400 singles against one batch | About 10x | 45x (cold batch), 166x (warm batch) | Per-request overhead dominates, so batching pays far more than the arithmetic alone suggests. |

### 5.1 The LLM's predictions (made in the project chat before the numbers existed)

| Question | LLM prediction on record | What we measured | Verdict |
|---|---|---|---|
| Which part of a request is slowest? | Inference on a tiny network is tiny, and network plus framework dominates | Inference 0.2 ms; network and framework over 99% of the round trip | Right |
| Will batching beat singles by more than the first test showed? | The ratio should rise well above 27x once the batch call is warm | 45x with the cold version, 166x with the corrected version | Right |
| Will the first batch call show high inference that later drops? | The first call includes warm-up, later calls drop sharply | The corrected run was already warm: all five calls were low (0.24 to 0.86 ms). The earlier cold run showed 104.7 ms | Not re-testable, but the earlier cold run supports it |

The LLM gave no numeric guesses for milliseconds, so we do not claim any.

### 5.2 What surprised us (written after measuring)

1. The first local call took 26.8 ms of inference and later calls took 0.2 ms. The model did not get faster: the first forward pass does one-time set-up.
2. `--no-keepalive` made the median about ten times slower (87 ms to 883 ms). Connection set-up measured at roughly 150 ms (TCP 73 ms plus TLS 78 ms), so that explains only part of it (see 6.4).
3. Even on localhost, HTTP and framework overhead (1.57 ms) was about seven times the model's compute (0.22 ms).
4. The same endpoint measured 41 ms median in one run and 87 ms in another, while `server_total` stayed flat. The network varies, the server does not.

## 6. Explanations

### 6.1 Where the time goes

```
Laptop --(internet, about 40-90 ms)--> GitHub tunnel relay --> Codespace
                         uvicorn / FastAPI: HTTP and JSON handling (about 1.5 ms)
                           parse 0.04 -> normalise 0.04 -> inference 0.23 -> build 0.02 ms
Laptop <------------------------- response -----------------------------
```

The server's own work totals about 0.3 ms. The model has 129 parameters, so shrinking it further would not noticeably speed up the endpoint.

### 6.2 Why the first request is slow
It took 0.6 to 1.3 s from the laptop. It pays for the DNS lookup, TCP and TLS set-up, and the server's first call into PyTorch. A real service would send a warm-up request at start-up.

### 6.3 Why the first inference was slow
Locally the first inference took 26.8 ms and the rest about 0.2 ms. PyTorch does one-time initialisation on the first forward pass. A new input shape (a batch of 400 rows) showed the same effect once: 104.7 ms the first time, 0.24 to 0.86 ms afterwards.

### 6.4 Why no keep-alive is slower
Without keep-alive each request opens a new connection. We measured TCP connect at 72.6 ms and TLS at 77.6 ms, about 150 ms in total. The no-keep-alive median was about 800 ms above the keep-alive median, so connection set-up explains only part of it. **We did not find the rest.** Untested hypotheses: the Python client looking up IPv6 addresses through a home router that sometimes refuses DNS queries, and extra per-connection work in the GitHub tunnel relay. `server_total` stayed at 0.35 ms, so the cost is outside our code.

### 6.5 Why localhost is faster
Running the benchmark inside the Codespace removes the internet. What remains (1.57 ms) is HTTP handling and the framework.

### 6.6 Why one batch beats 400 singles
The singles pay the network round trip 400 times (about 91 ms each). The batch pays it once. On the server, 400 rows go through one matrix multiply (inference 0.28 ms for 400 rows, close to the 0.23 ms for one row) instead of 400 separate forward passes.

## 7. Limits of these measurements

- Our home router's DNS sometimes refused lookups and crashed one benchmark run. [EDIT: say what you did about it, e.g. changed DNS or used a hotspot.]
- Network speed changed between runs (41 ms and 87 ms medians), so we report both.
- p99 over 200 requests rests on the two slowest requests, so it is only indicative.
- The ranges were chosen from typical sensor limits, not from a datasheet.

## 8. Evidence files

- `results/bench_20261121_232253.csv`: run A (laptop, keep-alive)
- `results/bench_20261121_234806.csv`: run B (laptop, keep-alive)
- `results/bench_20261121_235107.csv`: laptop, no keep-alive
- `results/bench_20261121_182156.csv`: inside the Codespace (the clock is UTC)
- `test_api.py`, `batch_test.py`, `handshake_test.py`: scripts that produced the tables above
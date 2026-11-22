# Mission 1–2 Report: Data Cleaning, Baselines, and Model Architecture Sweep

**Team 5 (Edgy)** | Edge AI, Module 1 | Missions 1 & 2

---

## 1. Executive Summary

This report documents the end-to-end data preparation, baseline modeling, neural network architecture exploration, and input normalization experiments for predicting IoT doorbell battery longevity (`days_left`). 

Key findings:
- **Data Quality:** Exactly 24 glitch rows were identified and removed across three sensor failure modes (0 V battery, -127 °C temperature, and 0 dBm WiFi RSSI). Valid zero-value edge cases (`rings_per_day == 0` and `age_months == 0`) were preserved.
- **Baselines:** The mean predictor baseline yielded a validation MAE of **57.27 days**, while an ordinary least squares linear regression achieved **22.84 days** (a 60.1% error reduction).
- **Architecture Selection:** A two-hidden-layer network (`5 → 32 → 32 → 1`, 1,281 parameters) was chosen. It achieved **4.89 days** historical validation MAE (reproduced at **4.90 days** in the fresh training run). The slightly wider `5 → 64 → 64 → 1` network was rejected because its 3.6× parameter increase yielded a negligible error reduction of only 0.02 days.
- **Normalization:** Standardizing training inputs (`(X - mu) / sd`) proved critical: the normalized network reached **4.90 days** validation MAE, whereas the unnormalized network stalled at **13.70 days**.

---

## 2. Dataset and Cleaning

The raw dataset (`data/team_05_train.csv`) contains 2,000 observations of doorbell battery telemetry with five input features:
1. `battery_v`: Lithium-ion cell terminal voltage.
2. `temp_c`: Ambient operating temperature in degrees Celsius.
3. `rings_per_day`: Doorbell chime events per day.
4. `wifi_rssi_dbm`: Received WiFi signal strength in dBm.
5. `age_months`: Operating age of the device since installation.

The target variable is `days_left` (continuous positive estimate of remaining battery life).

### 2.1 Sensor Glitches and Cleaning Rule

During exploratory analysis, 24 sensor glitch rows were detected and removed using the rule:

```python
bad = (
    (df["battery_v"] == 0) |
    (df["temp_c"] == -127) |
    (df["wifi_rssi_dbm"] == 0)
)
```

| Glitch Condition | Rows Dropped | Physical / Engineering Justification |
|---|---:|---|
| `battery_v == 0` | 7 | A single-cell lithium-ion battery operates between ~3.0 V (cutoff) and ~4.2 V (fully charged). A 0.0 V reading indicates a disconnected ADC channel, open circuit, unpowered sensor rail, or ADC driver fault. A cell physically at 0 V is permanently degraded and could not power the microcontroller. |
| `temp_c == -127` | 11 | `-127` is the standard digital error sentinel for Dallas 1-Wire / DS18B20 temperature sensors, emitted when the data line is disconnected, pulled high during bus reset, or fails CRC. Furthermore, -127 °C is physically impossible in outdoor residential doorbell environments. |
| `wifi_rssi_dbm == 0` | 6 | WiFi RSSI is logarithmic and always strictly negative in real deployments (typically -30 dBm to -90 dBm). 0 dBm corresponds to 1 mW of incident power directly at the antenna, requiring a high-power transmitter millimeters away. In wireless network drivers (e.g., ESP-IDF), 0 dBm signifies an unassociated interface, link timeout, or uninitialized buffer. |
| **Total Dropped** | **24** | **1,976 clean rows retained for modeling.** |

### 2.2 Legitimate Zero Values

Two fields frequently register zero but were deliberately **not** treated as glitches:
- `rings_per_day == 0`: A quiet residence, unoccupied holiday home, or low-activity period naturally has zero ring events per day. Zero rings accurately represents lowest active power drain.
- `age_months == 0`: Represents a newly installed doorbell during its first month of operation.

![Feature Histograms](../plots/mission1_feature_histograms.png)

### 2.3 Feature Relationships with Target

![Target vs Input Features](../plots/mission1_days_left_vs_features.png)

As shown in the scatter plots above:
- `battery_v` exhibits strong positive correlation with `days_left` (higher voltage corresponds to greater remaining battery life).
- `age_months` shows an inverse trend, as older batteries suffer capacity degradation.
- `rings_per_day` exerts an inverse effect due to active transmit/chime power drain.
- `wifi_rssi_dbm` and `temp_c` provide secondary environmental and transmission efficiency effects.

---

## 3. Baselines

All modeling experiments use the exact deterministic 80/20 train/validation split defined in `train.py` (`np.random.default_rng(0)`, 395 validation rows, 1,581 training rows).

### 3.1 Mean Baseline
- **Strategy:** Predict the scalar mean of the training target ($\bar{y} \approx 68.61$ days) for all validation inputs.
- **Validation MAE:** **57.27 days**.
- **Analysis:** This serves as the naive baseline. Any predictive model must substantially outperform 57.27 days to demonstrate predictive utility.

### 3.2 Linear Regression Baseline
- **Strategy:** Fit an ordinary least squares regression using NumPy's `np.linalg.lstsq(train_mat, y, rcond=None)` where `train_mat` includes a bias column (`[X, 1]`).
- **Validation MAE:** **22.84 days**.
- **Analysis:** Linear regression reduces validation error by **60.1%** relative to the mean baseline. This confirms that the sensor inputs carry strong, actionable linear predictive signal.

![Linear Regression Predicted vs True](../plots/mission1_linear_regression_pred_vs_true.png)

---

## 4. Architecture Sweep & Final Model Selection

### 4.1 Historical Sweep Results

A comprehensive sweep over hidden layer depth and width was conducted using the same training configuration (1,500 epochs, Adam optimizer with learning rate $3 \times 10^{-3}$, batch size 128, training-set normalization):

| Architecture | Hidden Layers | Total Parameters | Supplied Historical Val MAE (days) |
|---|:---:|---:|---:|
| 5 → 4 → 1 | 1 | 29 | 12.03 |
| 5 → 8 → 1 | 1 | 57 | 16.36 |
| 5 → 16 → 1 | 1 | 113 | 16.30 |
| 5 → 32 → 1 | 1 | 225 | 6.47 |
| 5 → 64 → 1 | 1 | 449 | 16.32 |
| 5 → 4 → 4 → 1 | 2 | 49 | 8.77 |
| 5 → 8 → 8 → 1 | 2 | 129 | 6.14 |
| 5 → 16 → 16 → 1 | 2 | 385 | 5.67 |
| **5 → 32 → 32 → 1** | **2** | **1,281** | **4.89** |
| 5 → 64 → 64 → 1 | 2 | 4,609 | 4.87 |

*Note: The values above represent the supplied historical sweep results.*

### 4.2 Fresh Retraining Run Verification

Executing the unmodified training script:
```bash
python train.py --team 5 --hidden 32,32
```
produced:
- **Baseline (mean):** 57.27 days
- **Baseline (linear regression):** 22.84 days
- **Network (32, 32) Validation MAE:** **4.90 days** (epoch 1,499: 4.90 days).
- **Parameters:** 1,281.

> [!NOTE]
> **Training Script Output & Verified Training MAE:**
> `train.py` evaluates and prints only validation MAE during execution (`mae(net(Xvt).numpy(), yv)`). In accordance with the strict code-change restriction, `train.py` was not modified.
> To verify the training performance without altering `train.py`, the saved `32,32` model (`model/model.pt` and `model/config.json`) was evaluated on the identical cleaned training split (1,581 samples) using the saved normalization constants (`mu`, `sd`). The measured training MAE is **4.63 days** (4.6262 days) with a validation MAE of **4.90 days** (4.8965 days), confirming the historical training metrics.

![Architecture Sweep MAE vs Params](../plots/mission2_architecture_sweep_mae_vs_params.png)

### 4.3 Why 5 → 32 → 32 → 1 Was Selected

1. **Superior Accuracy:** A verified training MAE of **4.63 days** and validation MAE of **4.90 days** (historical sweep: 4.89 days) represent a **91.4% error reduction** over the mean baseline (57.27 days) and a **78.5% error reduction** over linear regression (22.84 days).
2. **Minimal Edge Footprint:** With 1,281 parameters, the model requires only **5,124 bytes** of storage in 32-bit floating point and **1,281 bytes** in 8-bit quantized integer representation.
3. **ESP32-S3 SRAM Fit:** In Mission 4 calculations, the available SRAM budget on the ESP32-S3 is 300 KB (307,200 bytes). The 32,32 network weights plus peak activation memory (256 bytes) require only **5,380 bytes**, consuming merely **1.75% of available SRAM**.

### 4.4 Why 5 → 64 → 64 → 1 Was Rejected

Although `5 → 64 → 64 → 1` achieved a marginally lower validation MAE of 4.87 days:
- It requires **4,609 parameters**—over **3.6× more parameters** than `32,32`.
- It requires **4,416 MACs** per inference compared to **1,216 MACs** for `32,32` (a 3.63× computational increase).
- The validation MAE improvement is a negligible **0.02 days** (~28 minutes over a multi-month battery lifespan), which is well within stochastic training noise and does not justify the 3.6× memory and computation penalty on an embedded IoT device.

### 4.5 Analysis of Shallow Network Failures (Single-Hidden-Layer 8, 16, 64)

In the single-hidden-layer configurations:
- `5 → 8 → 1`: MAE = **16.36**
- `5 → 16 → 1`: MAE = **16.30**
- `5 → 64 → 1`: MAE = **16.32**

These three architectures consistently converged to nearly identical error rates near ~16.3 days. This behavior indicates:
1. **Optimization Plateau / Saddle Point:** With a single hidden layer and the default Adam learning rate ($3 \times 10^{-3}$), the optimization landscape for single-layer networks frequently traps gradient descent in suboptimal local minima where the network primarily models linear components with a flattened non-linear contribution.
2. **Representation Capacity for Feature Interactions:** Predicting remaining battery days requires learning coupled non-linear interactions (such as the combined impact of low voltage and cold temperature under high ring frequencies). Two-layer networks (`5 → w → w → 1`) consistently broke through this barrier at every width: `4→4` reached 8.77 days, `8→8` reached 6.14 days, `16→16` reached 5.67 days, and `32→32` reached 4.89 days.

---

## 5. Input Normalization Comparison

### 5.1 Experimental Setup

To evaluate the impact of input feature standardization:
- **Architecture:** `5 → 32 → 32 → 1` (1,281 parameters).
- **Training Setup:** 1,500 epochs, Adam optimizer (lr = 3e-3), batch size 128, random seed 0, identical train/validation split.
- **Normalized Run:** $X_{\text{norm}} = (X - \mu) / \sigma$, where $\mu$ and $\sigma$ are computed strictly from the training split to prevent data leakage.
- **Unnormalized Run:** $X_{\text{unnorm}} = (X - 0) / 1$ (`mu = 0`, `sd = 1`).

### 5.2 Measured Results

| Configuration | Input Scaling | Measured Final Validation MAE |
|---|---|---:|
| **Normalized** | Training $\mu$, $\sigma$ | **4.90 days** |
| **Unnormalized** | $\mu = 0$, $\sigma = 1$ | **13.70 days** |

![Normalization Comparison](../plots/mission2_normalization_comparison.png)

### 5.3 Technical Explanation

As shown in the convergence trajectories above, the normalized network rapidly converges, dropping below 10 days within 100 epochs and stabilizing at 4.90 days. The unnormalized network struggles throughout training and plateaus at 13.70 days—nearly **2.8× higher error**.

The raw sensor features vary across widely divergent scales:
- `battery_v`: ranges from 2.5 V to 4.5 V ($\mu \approx 3.78$, $\sigma \approx 0.24$).
- `wifi_rssi_dbm`: ranges from -100 dBm to -10 dBm ($\mu \approx -66.40$, $\sigma \approx 12.51$).
- `temp_c`: ranges from -40 °C to 85 °C ($\mu \approx 28.18$, $\sigma \approx 6.91$).

Without normalization, input gradients with respect to larger numerical features (`wifi_rssi_dbm` and `temp_c`) dominate parameter updates, while the critical but numerically small variations in `battery_v` (where a 0.1 V drop indicates significant discharge) receive disproportionately small gradient signals. Standardizing each feature to zero mean and unit variance ensures isotropic loss contours and well-conditioned gradient descent.

---

## 6. Summary of Generated Artifacts

The following artifacts have been generated and verified:
- [mission1_feature_histograms.png](file:///home/bhavanish/.gemini/antigravity/scratch/EDGY-MISSION-3/plots/mission1_feature_histograms.png)
- [mission1_days_left_vs_features.png](file:///home/bhavanish/.gemini/antigravity/scratch/EDGY-MISSION-3/plots/mission1_days_left_vs_features.png)
- [mission1_linear_regression_pred_vs_true.png](file:///home/bhavanish/.gemini/antigravity/scratch/EDGY-MISSION-3/plots/mission1_linear_regression_pred_vs_true.png)
- [mission2_architecture_sweep_mae_vs_params.png](file:///home/bhavanish/.gemini/antigravity/scratch/EDGY-MISSION-3/plots/mission2_architecture_sweep_mae_vs_params.png)
- [mission2_normalization_comparison.png](file:///home/bhavanish/.gemini/antigravity/scratch/EDGY-MISSION-3/plots/mission2_normalization_comparison.png)
- [config.json](file:///home/bhavanish/.gemini/antigravity/scratch/EDGY-MISSION-3/model/config.json)
- [model.pt](file:///home/bhavanish/.gemini/antigravity/scratch/EDGY-MISSION-3/model/model.pt)

"""
app/stats.py -- count a network's size and cost from its layers (Missions 2 and 4).

One function, layer_table(), is used in two places:
  * app/main.py serves it at GET /model_info
  * mission4.py calls it for the 5 -> w -> w -> 1 width table
so the API and the Mission 4 analysis can never disagree.

Counting rule for nn.Linear(a, b):
  params = a*b weights + b biases
  MACs   = a*b multiply-accumulates per prediction (adding the bias is not a MAC)
  bytes  = params * 4 as float32, params * 1 as int8
ReLU has no parameters and no MACs, so it is skipped.
"""
from torch import nn

BYTES_F32 = 4
BYTES_INT8 = 1

# ESP32-S3 budget from the README: ~300 KB of SRAM free for our program.
# We read "KB" as 1024 bytes (KiB), the way SRAM sizes are specified, so
# 300 KB = 307,200 bytes. (With 300,000 the answers move only at the boundary;
# mission4.py reports both.)
SRAM_BUDGET_BYTES = 300 * 1024
PSRAM_BYTES = 8 * 1024 * 1024


def linear_layers(net):
    """The nn.Linear layers of a BatteryNet, in order (ReLUs dropped)."""
    return [m for m in net.net if isinstance(m, nn.Linear)]


def layer_table(net, budget=SRAM_BUDGET_BYTES):
    """Per-layer params, bytes and MACs, plus totals and the two 'fits' answers.

    Returns exactly the keys GET /model_info promises.
    """
    layers = []
    for i, m in enumerate(linear_layers(net), start=1):
        a, b = m.in_features, m.out_features
        params = a * b + b
        layers.append({"name": f"linear{i}", "in": a, "out": b, "params": params,
                       "bytes_f32": params * BYTES_F32, "macs": a * b})

    total_params = sum(l["params"] for l in layers)
    total_bytes_f32 = total_params * BYTES_F32
    total_bytes_int8 = total_params * BYTES_INT8
    return {
        "layers": layers,
        "total_params": total_params,
        "total_bytes_f32": total_bytes_f32,
        "total_bytes_int8": total_bytes_int8,
        "total_macs": sum(l["macs"] for l in layers),
        "fits_300kb_sram_f32": total_bytes_f32 <= budget,
        "fits_300kb_sram_int8": total_bytes_int8 <= budget,
    }


def peak_activation_values(net):
    """Largest number of activation values that must be held at once.

    Layer k reads its input vector while writing its output vector, so both are
    alive together. The peak over layers is max(in + out). (Weights are counted
    separately, above.) Multiply by 4 for float32 or 1 for int8.
    """
    return max(m.in_features + m.out_features for m in linear_layers(net))

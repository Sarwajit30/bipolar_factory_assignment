"""Part 1: layer-by-layer profile of ResNet-18.

For each block we record:
  MACs, parameter memory, output activation bytes (FP32 and INT8),
  and measured latency (mean and std over many runs, after warm-up).
"""
import csv
import time

import torch
import torch.nn as nn
import torchvision

WARMUP = 10     # throwaway runs, the first runs are slower for random reasons
RUNS = 50       # measured runs per block
torch.set_num_threads(1)   # one CPU thread, so timings are stable and comparable

# ---------- 1. Load the model and cut it into blocks ----------
model = torchvision.models.resnet18(
    weights=torchvision.models.ResNet18_Weights.DEFAULT
).eval()

blocks = [
    ("stem (conv1+bn+relu)", nn.Sequential(model.conv1, model.bn1, model.relu)),
    ("maxpool", model.maxpool),
]
for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
    for i, blk in enumerate(getattr(model, layer_name)):
        blocks.append((f"{layer_name}.{i}", blk))
blocks.append(("avgpool+flatten", nn.Sequential(model.avgpool, nn.Flatten())))
blocks.append(("fc", model.fc))


# ---------- 2. Helpers ----------
def count_macs(module, x):
    """Count multiply-accumulates by hooking every Conv2d and Linear inside module."""
    total = [0]

    def conv_hook(m, inp, out):
        kh, kw = m.kernel_size
        total[0] += out.numel() * (m.in_channels // m.groups) * kh * kw

    def linear_hook(m, inp, out):
        total[0] += m.in_features * m.out_features * (out.numel() // m.out_features)

    hooks = []
    for sub in module.modules():
        if isinstance(sub, nn.Conv2d):
            hooks.append(sub.register_forward_hook(conv_hook))
        elif isinstance(sub, nn.Linear):
            hooks.append(sub.register_forward_hook(linear_hook))
    with torch.no_grad():
        module(x)
    for h in hooks:
        h.remove()
    return total[0]


def time_block(module, x):
    """Return (mean_ms, std_ms) of one forward pass of module on input x."""
    with torch.no_grad():
        for _ in range(WARMUP):
            module(x)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            module(x)
            times.append((time.perf_counter() - t0) * 1000)
    t = torch.tensor(times)
    return t.mean().item(), t.std().item()


# ---------- 3. Profile ----------
x = torch.randn(1, 3, 224, 224)   # one fake image, same size as a real one
rows = []

# Row 0: the raw input. Sending this is what "no split" costs.
rows.append({
    "block": "input image", "MMACs": 0, "param_KB": 0,
    "out_shape": str(tuple(x.shape)),
    "fp32_KB": x.numel() * 4 / 1024, "int8_KB": x.numel() * 1 / 1024,
    "lat_mean_ms": 0, "lat_std_ms": 0,
})

for name, blk in blocks:
    macs = count_macs(blk, x)
    params = sum(p.numel() for p in blk.parameters()) * 4   # FP32 bytes
    mean_ms, std_ms = time_block(blk, x)
    with torch.no_grad():
        out = blk(x)
    rows.append({
        "block": name, "MMACs": macs / 1e6, "param_KB": params / 1024,
        "out_shape": str(tuple(out.shape)),
        "fp32_KB": out.numel() * 4 / 1024, "int8_KB": out.numel() * 1 / 1024,
        "lat_mean_ms": mean_ms, "lat_std_ms": std_ms,
    })
    x = out   # output of this block is the input of the next

# ---------- 4. Whole-model baseline (measured separately, fair comparison) ----------
full_in = torch.randn(1, 3, 224, 224)
full_mean, full_std = time_block(model, full_in)

# ---------- 5. Print and save ----------
hdr = f"{'block':24}{'MMACs':>9}{'paramKB':>10}{'out_shape':>20}{'fp32KB':>10}{'int8KB':>9}{'ms':>8}{'std':>7}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['block']:24}{r['MMACs']:9.1f}{r['param_KB']:10.1f}{r['out_shape']:>20}"
          f"{r['fp32_KB']:10.1f}{r['int8_KB']:9.1f}{r['lat_mean_ms']:8.2f}{r['lat_std_ms']:7.2f}")
print(f"\nSum of block latencies: {sum(r['lat_mean_ms'] for r in rows):.2f} ms")
print(f"Whole model in one go:  {full_mean:.2f} +- {full_std:.2f} ms")

with open("part1_profile.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print("Saved part1_profile.csv")

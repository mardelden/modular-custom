#!/usr/bin/env python3
"""Build the fp8-attention gallery: copy baseline/qkv/full renders and generate
legible diff-overlay heatmaps (dimmed gray image + hot-colored difference along
edges/lines), into an output dir. Prints a DELTAS JS literal for the viewer.

Run on max-build:
  /root/wheeltest-baked/bin/python /root/fp8sim_gallery.py
"""
import os
import shutil

import numpy as np
from PIL import Image

BASE_DIR = "/root/w4a4_naive_kernel_renders"   # baseline W4A4 (bf16 attention)
SIM_DIR = "/root/fp8sim"                        # qkv_/full_ renders
OUT = "/root/fp8sim/gallery"
NAMES = ["portrait", "text", "macro", "dark"]
MODES = ["qkv", "full"]
SCALE = 40.0     # diff magnitude (0..255) that saturates the heatmap
GRAYDIM = 0.28   # how faint the baseline shows under the heatmap

os.makedirs(OUT, exist_ok=True)


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)


def save(arr, p):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").save(p)


def diff_overlay(a, b):
    """Dimmed grayscale baseline with the |a-b| magnitude as a hot heatmap.

    Unchanged pixels show the faint gray image; differences glow red->yellow->
    white along the edges/lines where the two renders diverge.
    """
    dmag = np.abs(a - b).mean(axis=2)          # [H,W] 0..255
    t = np.clip(dmag / SCALE, 0.0, 1.0)
    hr = np.clip(t / 0.40, 0, 1)
    hg = np.clip((t - 0.40) / 0.40, 0, 1)
    hb = np.clip((t - 0.80) / 0.20, 0, 1)
    gray = (0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]) * GRAYDIM
    out = np.stack([gray + hr * 255, gray + hg * 255, gray + hb * 255], axis=2)
    return out


def metrics(a, b):
    d = a - b
    mae = float(np.abs(d).mean())
    rmse = float(np.sqrt((d * d).mean()))
    return mae, rmse


deltas = {}
for name in NAMES:
    base_p = os.path.join(BASE_DIR, f"w4a4_{name}.png")
    a = load(base_p)
    shutil.copy(base_p, os.path.join(OUT, f"w4a4_{name}.png"))
    deltas[name] = {}
    for mode in MODES:
        sim_p = os.path.join(SIM_DIR, f"{mode}_{name}.png")
        b = load(sim_p)
        shutil.copy(sim_p, os.path.join(OUT, f"{mode}_{name}.png"))
        save(diff_overlay(a, b), os.path.join(OUT, f"{mode}diff_{name}.png"))
        mae, rmse = metrics(a, b)
        deltas[name][mode] = (mae, rmse)

# emit the JS DELTAS literal
print("const DELTAS = {")
for name in NAMES:
    q = deltas[name]["qkv"]
    f = deltas[name]["full"]
    print(
        f"  {name+':':10}"
        f"{{qkv:{{mae:{q[0]:.2f}, rmse:{q[1]:.2f}}}, "
        f"full:{{mae:{f[0]:.2f}, rmse:{f[1]:.2f}}}}},"
    )
print("};")
print(f"\nWROTE gallery -> {OUT} ({len(os.listdir(OUT))} files)")

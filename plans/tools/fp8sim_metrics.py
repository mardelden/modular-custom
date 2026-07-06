#!/usr/bin/env python3
"""Phase Q quality metrics: compare fp8-attention SIM renders vs the known-good
W4A4 renders. Prints a MAE/RMSE/PSNR table and writes a self-contained
compare.html (baseline | each mode | amplified-diff, per prompt).

Run on max-build with the baked venv python (has numpy + PIL):
  /root/wheeltest-baked/bin/python /root/fp8sim_metrics.py qkv full
"""
import base64
import io
import os
import sys

import numpy as np
from PIL import Image

BASELINE_DIR = "/root/w4a4_naive_kernel_renders"
SIM_DIR = "/root/fp8sim"
NAMES = ["portrait", "text", "macro", "dark"]
MODES = sys.argv[1:] or ["qkv", "full"]
OUT_HTML = os.path.join(SIM_DIR, "compare.html")
DIFF_AMP = 10.0  # amplify abs pixel diff for the visualization


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def metrics(a, b):
    d = a - b
    mae = float(np.abs(d).mean())
    rmse = float(np.sqrt((d * d).mean()))
    maxd = float(np.abs(d).max())
    mse = float((d * d).mean())
    psnr = float("inf") if mse == 0 else 20 * np.log10(255.0) - 10 * np.log10(mse)
    return mae, rmse, maxd, psnr


def png_b64_from_path(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def png_b64_from_array(arr_uint8):
    im = Image.fromarray(arr_uint8, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def img_tag(b64, w=340):
    return f'<img src="data:image/png;base64,{b64}" width="{w}">'


# ---- compute ----
results = {}  # (mode, name) -> (mae, rmse, maxd, psnr, diff_b64)
print(
    f"{'mode':6} {'prompt':10} {'MAE':>8} {'RMSE':>8} {'maxD':>6} {'PSNR(dB)':>9}"
)
print("-" * 52)
for mode in MODES:
    maes = []
    for name in NAMES:
        base_p = os.path.join(BASELINE_DIR, f"w4a4_{name}.png")
        sim_p = os.path.join(SIM_DIR, f"{mode}_{name}.png")
        if not (os.path.exists(base_p) and os.path.exists(sim_p)):
            print(
                f"{mode:6} {name:10}  MISSING "
                f"(base={os.path.exists(base_p)} sim={os.path.exists(sim_p)})"
            )
            continue
        a, b = load(base_p), load(sim_p)
        if a.shape != b.shape:
            print(f"{mode:6} {name:10}  SHAPE MISMATCH {a.shape} vs {b.shape}")
            continue
        mae, rmse, maxd, psnr = metrics(a, b)
        maes.append(mae)
        diff = np.clip(np.abs(a - b) * DIFF_AMP, 0, 255).astype(np.uint8)
        results[(mode, name)] = (mae, rmse, maxd, psnr, png_b64_from_array(diff))
        print(
            f"{mode:6} {name:10} {mae:8.3f} {rmse:8.3f} {maxd:6.0f} {psnr:9.2f}"
        )
    if maes:
        print(f"{mode:6} {'MEAN':10} {np.mean(maes):8.3f}")
        print("-" * 52)

# ---- HTML ----
base_b64 = {n: png_b64_from_path(os.path.join(BASELINE_DIR, f"w4a4_{n}.png"))
            for n in NAMES
            if os.path.exists(os.path.join(BASELINE_DIR, f"w4a4_{n}.png"))}

parts = [
    "<!doctype html><meta charset=utf-8>",
    "<title>fp8-attn SIM vs W4A4</title>",
    "<style>body{font-family:sans-serif;background:#111;color:#eee;margin:16px}"
    "table{border-collapse:collapse}td{border:1px solid #333;padding:6px;"
    "vertical-align:top;text-align:center}th{padding:6px}"
    ".m{font-size:12px;color:#9cf;font-family:monospace}"
    "h2{margin:18px 0 6px}</style>",
    "<h1>fp8-attention simulation vs known-good W4A4 render</h1>",
    f"<p>baseline = W4A4 (bf16 attention). diff images amplified &times;{DIFF_AMP:.0f}. "
    "Lower MAE / higher PSNR = closer to baseline.</p>",
]

for name in NAMES:
    if name not in base_b64:
        continue
    parts.append(f"<h2>{name}</h2>")
    parts.append("<table><tr>")
    parts.append("<th>W4A4 baseline</th>")
    for mode in MODES:
        parts.append(f"<th>{mode}</th><th>{mode} diff&times;{DIFF_AMP:.0f}</th>")
    parts.append("</tr><tr>")
    parts.append(f"<td>{img_tag(base_b64[name])}</td>")
    for mode in MODES:
        key = (mode, name)
        sim_p = os.path.join(SIM_DIR, f"{mode}_{name}.png")
        if key in results and os.path.exists(sim_p):
            mae, rmse, maxd, psnr, diff_b64 = results[key]
            simg = png_b64_from_path(sim_p)
            parts.append(
                f"<td>{img_tag(simg)}<div class=m>MAE {mae:.2f}<br>"
                f"RMSE {rmse:.2f}<br>PSNR {psnr:.1f} dB<br>maxΔ {maxd:.0f}</div></td>"
            )
            parts.append(f"<td>{img_tag(diff_b64)}</td>")
        else:
            parts.append("<td>(missing)</td><td>(missing)</td>")
    parts.append("</tr></table>")

with open(OUT_HTML, "w") as f:
    f.write("\n".join(parts))
print(f"\nWROTE {OUT_HTML}")

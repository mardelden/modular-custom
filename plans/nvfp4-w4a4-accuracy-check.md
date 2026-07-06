# NVFP4 W4A4 (fp4-activation) image-quality pre-check — results & tooling

> Date: 2026-07-05. Purpose: before committing the multi-week native-FP4 (W4A4) GEMM,
> cheaply measure whether quantizing **activations** to fp4 (1 mantissa bit) degrades the
> render. Method: inject a **faithful fp4-activation fake-quant** (bf16→fp4→bf16) into the
> existing W4A16 path (no kernel needed) and compare renders. Reusable for future
> quantization-accuracy screens.

## Verdict (this render): W4A4 is GO-leaning — no visible degradation, real bounded error
- The fake-quant is **faithful** (injects a real 9.15% fp4 activation error, exact e2m1
  snapping — not a no-op).
- W4A4-sim vs W4A16 render: **measurable but modest** (MAE 7.3 / RMSE 18.0 of 255), **no
  gross corruption**, and **visually indistinguishable quality** (same vase/tulips/table,
  no washout/artifacts/color-shift).
- **Caveat:** ONE prompt, one seed, a simple scene. fp4 activations bite hardest on
  faces/hands, text, fine texture, dark/complex scenes — **run harder prompts (below)
  before the GEMM commit.**

## The numbers the sim calculated (faithfulness check, ran on sm_120 GPU)
`/root/fp4_faithfulness_check.py` (saved locally at `plans/tools/fp4_faithfulness_check.py`):
- `q` values **all land exactly on the e2m1 grid** {0,±.5,±1,±1.5,±2,±3,±4,±6}.
- exact dequant identity `x_hat_f32 == q·scale·input_scale`: **err 0.00e+00**.
- exact e2m1 grid-snap err: **4.77e-07** (≈0).
- block scales: all valid `float8_e4m3fn`.
- **reconstruction rel error `‖x_hat − x‖/‖x‖ = 9.15%`** → real, bounded fp4 activation error.
- bf16 output cast err 1.25e-2 (expected — output is bf16 for the matmul).

## Render results (seed 42, prompt "a red vase with yellow tulips on a wooden table, studio photo", 1024²)
| image | file | delta |
|---|---|---|
| W4A16 known-good | `/root/klein_nvfp4_fixed.png` (1,463,688 B) | reference |
| W4A16 baseline repro (flag OFF) | `/root/klein_w4a16_repro.png` (1,463,688 B) | **byte-identical, MAE 0.000** → harness valid, gated edit inert when off |
| **W4A4-sim** (MODULAR_SIM_FP4_ACT=1) | `/root/klein_w4a4_sim.png` (1,466,612 B) | vs W4A16: **MAE 7.3 / RMSE 18.0**; vs bf16: MAE 10.5 / RMSE 22.3 |
| bf16 known-good | `/root/klein_bf16.png` (1,500,537 B) | (different arrangement = weight-quant trajectory divergence, not quality) |
Local copies pulled to `/tmp/cmp_{w4a16,w4a4_sim,bf16}.png`.

## The faithful fake-quant (exact NVFP4 activation recipe) — reusable
Lives on max-build only, in `quant_ops.py`, env-gated `MODULAR_SIM_FP4_ACT=1` (OFF by
default), scoped to the `_is_cuda_fp4_gpu()` W4A16 branch. Implemented as
**hardware-agnostic graph ops** because `quantize_dynamic_block_scaled`
(`fp4_quantization.mojo:2070`) is **SM100-only** — it does not compile on sm_120.

```python
def _snap_to_e2m1(v):  # round onto signed E2M1 grid {0,.5,1,1.5,2,3,4,6}, clamp 6
    a = ops.abs(v); z = a * 0.0
    sign = ops.where(v < 0.0, z - 1.0, z + 1.0)
    mag = ops.where(a > 5.0, z + 6.0,
          ops.where(a >= 3.5, z + 4.0,
          ops.where(a >= 2.5, z + 3.0,
          ops.where(a >= 1.75, z + 2.0,
          ops.where(a >= 1.25, z + 1.5,
          ops.where(a >= 0.75, z + 1.0,
          ops.where(a >= 0.25, z + 0.5, z)))))))
    return sign * mag

def _fake_quant_fp4_activation(x, input_scale):  # bf16 -> fp4 -> bf16, per-block-16 along K
    K = int(x.shape[1]); n_blk = K // 16
    isc = input_scale.cast(DType.float32).to(x.device)
    xs = x.cast(DType.float32) / isc                    # native tensor_sf = 1/input_scale
    xb = xs.reshape([x.shape[0], n_blk, 16])
    amax = ops.max(ops.abs(xb), axis=-1)                # [M, n_blk, 1]
    scale_q = (amax / 6.0).cast(DType.float8_e4m3fn).cast(DType.float32)
    scale_safe = ops.where(scale_q > 0.0, scale_q, scale_q * 0.0 + 1.0)
    q = _snap_to_e2m1(xb / scale_safe)
    x_hat = (q * scale_q).reshape([x.shape[0], K]) * isc
    return x_hat.cast(DType.bfloat16)

# in _matmul_float4, first line of the `if _is_cuda_fp4_gpu():` branch:
#     if os.environ.get("MODULAR_SIM_FP4_ACT") == "1":
#         x = _fake_quant_fp4_activation(x, input_scale)
```
Backup of the original: `root@max-build:/root/quant_ops.py.bak`. NOTE: this fake-quant is a
QUALITY simulation (bf16 MMA on degraded activations); the real W4A4 needs the native FP4
GEMM (spike GO, `nvfp4_mma_spike.mojo`) + an sm_120 activation quantizer.

## Reproduce (max-build; uses the shared GPU — coordinate so it's free)
```bash
# W4A4-sim render:
MODULAR_SIM_FP4_ACT=1 bash /root/serve_klein_nvfp4.sh   # serves NVFP4 Klein from source; wait "Server ready"
/root/wheeltest-baked/bin/python /root/render_klein.py /root/out.png "PROMPT"   # seed 42, 1024²
# baseline (no sim): same without MODULAR_SIM_FP4_ACT.
# faithfulness check: LD_LIBRARY_PATH=/opt/nvidia-libs/nvidia/*/lib /root/wheeltest/bin/python /root/fp4_faithfulness_check.py
```

## Harder prompts to run next (stress fp4 activations before the GEMM commit)
For each: render W4A16 (flag off) vs W4A4-sim (flag on), compare (and vs bf16 if the
max-serve bf16 container is up). Look for: face/skin artifacts, garbled text, lost fine
detail, banding in dark/smooth gradients.
1. Portrait/skin: "a close-up portrait photo of an elderly woman, wrinkled skin, sharp detail, soft studio light"
2. Text: "a vintage neon storefront sign that reads 'OPEN 24 HOURS' at night, reflections on wet pavement"
3. Fine texture: "a macro photo of a dragonfly on a dew-covered spider web, intricate detail, shallow depth of field"
4. Dark/gradient: "a dimly lit gothic cathedral interior lit only by candlelight, deep shadows, volumetric light"

If these hold up like the vase render → **confident GO** for the native-FP4 GEMM (D2/M1).
If artifacts appear → pivot to **W4A8** (`mxf8f6f4`, fp8 act, safer) or selective-bf16 on
sensitive layers.

## RESULTS (2026-07-05) — all 4 hard prompts PASSED → **W4A4 quality is a GO**
Rendered W4A16 (baseline, flag off) vs W4A4-sim (`MODULAR_SIM_FP4_ACT=1`), seed 42, 1024²,
same prompts as the bf16 refs. Per-prompt pixel deltas (0–255) + visual verdict:

| prompt | w4a4sim vs w4a16 (the gate) MAE/RMSE | w4a16 vs bf16 (context) MAE/RMSE | visual |
|---|---|---|---|
| portrait | **3.71 / 5.88** | 6.66 / 10.91 | near-identical; skin/wrinkle detail intact |
| text | **14.71 / 26.84** | 20.28 / 33.38 | "OPEN 24 HOURS" fully legible; no garbling |
| macro | **11.12 / 22.40** | 13.49 / 24.68 | dragonfly wings + dew preserved; no blur |
| dark | **7.01 / 14.94** | 8.18 / 17.00 | smooth gradients; **no banding** |
| **mean** | **9.14 / 17.52** | **12.20 / 21.50** | |

**Key finding:** on EVERY prompt the fp4-activation delta (w4a4sim vs w4a16) is **smaller
than the weight-quant delta already in production** (w4a16 vs bf16). The W4A4 error is
spatially local (global mean/std barely shift, e.g. portrait 121.17→121.18) — "different
details," not "worse quality." No artifacts / garbling / banding / washout across the 4
classic fp4 failure modes (faces, legible text, fine detail, dark gradients). Verified the
sim was genuinely active (`MODULAR_SIM_FP4_ACT=1` in the worker `/proc/<pid>/environ`).

**VERDICT: W4A4 image quality holds → GO to build the native FP4 GEMM (D2/M1 + sm_120
activation quantizer).** Images: `/Users/mardel/nvfp4-w4a4-renders/{w4a16,w4a4sim,bf16}_<prompt>.png`.

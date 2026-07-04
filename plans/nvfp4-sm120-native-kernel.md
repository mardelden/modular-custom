# Plan: NVFP4 (W4A16) matmul for FLUX.2-Klein on sm_120 (RTX PRO 6000 Blackwell)

> Status: **Reshaped 2026-07-04** after discovering upstream already landed the
> generic W4A16 scaffolding. Target = correct NVFP4 on sm_120 with the VRAM win
> at ~bf16 speed (materialize→dense), with an optional later fused/native path
> for full FP4 throughput.
> Branch `feat/nvfp4-sm120-native-kernel` is **rebased onto upstream/main
> `d35568099c` (July 3 nightly, Mojo dev2026070306 / MAX 26.5.0.dev2026070306)**.
> Backup of the pre-rebase state: branch `backup/nvfp4-pre-rebase-20260704`.

## What changed vs the original plan

Upstream commit **`06e82346ae [Kernels][MAX] Apple M5: NVFP4 W4A16 for
FLUX.2-dev`** (Fabio Riccardi, merged Jul 2) already built the entire generic
weight-only (W4A16) seam we were going to write from scratch, and it
**side-steps the sm_120 rank-5 wall entirely**:

- **`_matmul_float4` W4A16 branch** (`max/python/max/nn/quant_ops.py:98`):
  activations stay **bf16** (NOT dynamically quantized to FP4), weight block
  scales are **plain rank-2 `[N, K//16]`** (NOT the SM100 rank-5 TCGEN05
  interleave — this is exactly the layout sm_120 could not produce), FP4 weight
  dequantized to bf16 in-register, and `weight_scale_2` folded as a post-matmul
  scalar multiply. Only the *guard* (`_is_apple_gpu()`) and the *kernel* are
  platform-specific; the seam is generic.
- **Portable dequant kernel** `fp4_materialize_kernel` /
  `enqueue_fp4_materialize` (`max/kernels/src/linalg/matmul/gpu/apple/
  fp4_dequant.mojo`): explicitly **"hardware-neutral — plain LUT + scale, no
  PTX / MFMA intrinsics"**. One thread per output element, `E2M1_TO_FLOAT32[nib]
  * |block_scale|`. **Reusable verbatim on sm_120/CUDA.**
- **materialize→dense launcher** `_enqueue_apple_fp4_materialize_dense`
  (`fp4_matmul.mojo:720`): transient `[N,K]` bf16 buffer ← `enqueue_fp4_
  materialize` ← packed FP4 + scales, then the platform dense bf16 GEMM. Its
  docstring names **`mxfp4_dequant_matmul_amd` as "the AMD W4A16 sibling"** — so
  this "portable materialize + platform dense GEMM" pattern already ships for
  **Apple and AMD**. NVIDIA/sm_120 is the missing third sibling.
- **Mixed-precision diffusion pipeline plumbing**: per-component encodings via
  `--model-override 'transformer.quantization_encoding=float4_e2m1fnx2'`, a
  `latents_in_dtype` cast at the transformer→VAE graph boundary
  (`vae_decoder.py`), and an f32→bf16 weight-path fallback
  (`lib/config/model_config.py:682`). This **supersedes our Phase-0 forced-bf16
  hacks and the 21 GB combined-repo** (use the base repo + `--model-override
  transformer.weight_path=<nvfp4 safetensors>` instead). Landed on
  `flux2_executor.py` (FLUX.2-**dev**) — must be ported to our
  `flux2_klein_executor.py` (FLUX.2-**Klein**).

**Consequence:** native sm_120 FP4 `mma.sync` PTX intrinsics — the multi-week
hard part of the old plan — are **no longer required** for a correct,
VRAM-saving, ~bf16-speed result. They become an optional later perf phase.

## Op / dispatch API to mirror (already in tree post-rebase)
- Op registration `mo.matmul.weight.only.block.scaled.apple`
  (`max/kernels/src/graph_compiler/builtin_kernels/linalg.mojo:1016`) — asserts
  `has_apple_gpu_accelerator()`. Model a `.cuda` sibling on it.
- Python wrapper `_apple_weight_only_block_scaled_matmul`
  (`max/python/max/nn/kernels.py:6278`); guard `_is_apple_gpu()`
  (`kernels.py:6469`). Model `_cuda_weight_only_block_scaled_matmul` +
  `_is_cuda_fp4_gpu()` on them.
- AMD sibling launcher to read for the NVIDIA dense-GEMM wiring:
  `mxfp4_dequant_matmul_amd` (grep `max/kernels/src/linalg`).

---

## Phase A — Mixed-precision loading via upstream mechanism (Python only, no build)
Replace our Phase-0 hacks with the upstream approach, ported to Klein.
- **Drop** the forced `encoding = "bfloat16"` in `flux2/components/
  vae_decoder.py` + `image_encoder.py` and the `_component_encoding` override in
  `flux2_klein_executor.py` (commit `a5bc2d85a2`). Rely on `--model-override`
  per-component encoding + the f32→bf16 fallback in `model_config.py`.
- **Port** the `latents_in_dtype` cast (transformer→VAE boundary) into the Klein
  executor's `VaeDecoder` construction (mirror `flux2_executor.py`'s
  `latents_in_dtype=self._model_dtype`). Confirm whether Klein's VAE compiles
  bf16 (then it's a no-op) or f32.
- **Verify** the `--model-override` path resolves for the Klein arch/executor
  (it was written for `flux2`/`flux2_executor`; Klein may need the same override
  plumbing). Serve base Klein repo + `--model-override transformer.
  quantization_encoding=float4_e2m1fnx2` + `transformer.weight_path=<nvfp4>`.
- Deploy: scp `.py` only (July-2 wheel is fine for Python-only). No combined
  repo needed. Loads NVFP4; matmul still errors on sm_120 until Phase C.

## Phase B — Host-dequant oracle (Python only, no build) — OPTIONAL
Now largely redundant: the dequant math is already covered by the Apple/AMD
W4A16 tests and the portable `fp4_materialize_kernel`. Keep as a zero-build
fallback if we need a pure-Python reference image before the kernel builds:
numpy-port `_dequantize_nvfp4_to_bf16` (`max/tests/integration/nn/ep/
test_ep_moe_fp4.py:58`) in `nvfp4_weight_adapter.py`, make nvfp4 layers plain
bf16 `Linear`. Generate the fixed prompt+seed reference image = correctness
oracle for Phase C.

## Phase C — CUDA/sm_120 W4A16 materialize→dense (first bazelw build) — THE MAIN TASK
Add the missing NVIDIA sibling to the existing W4A16 machinery.
1. **Mojo launcher** `_enqueue_cuda_fp4_materialize_dense` (new
   `max/kernels/src/linalg/matmul/gpu/.../fp4_matmul_cuda.mojo` or fold into an
   existing NVIDIA linalg file): transient `[N,K]` bf16 buffer ←
   `enqueue_fp4_materialize[bf16]` (portable, reuse as-is) ← packed FP4 +
   scales, then the **existing NVIDIA dense bf16 GEMM** (`linalg.matmul` /
   `matmul/gpu` multistage; find the entry point the AMD sibling uses). Same
   stream-ordered transient-buffer lifetime idiom (`_ = wdense_dev^`).
2. **Op registration** `mo.matmul.weight.only.block.scaled.cuda` in
   `builtin_kernels/linalg.mojo` — mirror `Struct_matmul_weight_only_block_
   scaled_apple`, assert a CUDA/NVIDIA accelerator instead of Apple.
3. **Python wrapper** `_cuda_weight_only_block_scaled_matmul` in `kernels.py`
   (mirror the Apple wrapper) + guard `_is_cuda_fp4_gpu()` (sm_120/sm_121; extend
   later). In `_matmul_float4` (`quant_ops.py:95`), add a branch **before** the
   Apple branch: `if _is_cuda_fp4_gpu(): res = _cuda_weight_only_block_scaled_
   matmul(...); return (res.f32 * weight_scale_2).bf16` — identical scalar fold.
4. **Deploy:** first kernel build — `./bazelw build //max/kernels/...`, deploy
   rebuilt kernel artifacts into the max-serve wheel's mojocache, **and bump the
   max-serve wheel to the July-3 nightly (`dev2026070306`)** so built kernels are
   ABI-matched to the runtime. Then scp the `.py`.
5. **Verify:** load NVFP4 Klein, generate fixed prompt+seed, pixel-compare to the
   Phase B oracle (or the bf16 baseline structurally). Confirm the VRAM drop
   (transformer weights stay 4-bit resident; the bf16 weight is a per-op
   transient). Result: correct NVFP4 + VRAM win at ~bf16 speed.

## Phase D — Fused / native sm_120 FP4 for full throughput — OPTIONAL, LATER
Only if Phase C's DRAM-bound materialize is too slow. Options, cheapest first:
1. **Fused in-register dequant** CUDA kernel (dequant B fragment in the GEMM
   loader seam, weight stays 4-bit in DRAM) — the CUDA analogue of Apple's fused
   `AppleM5Fp4MatMul` / the AMD fused path. No new PTX; reuses bf16 MMA.
2. **Native sm_120 block-scaled FP4 `mma.sync`** (`mma.sync...kind::mxf4nvf4.
   block_scale`, valid on `sm_120a`/`ptx87`) — the old Phase 3. Real FP4
   tensor-core throughput; writes new PTX intrinsics that don't exist in Mojo
   today (`mma_nvidia.mojo` has none; `KIND_MXF4NVF4` is SM100 UMMA only). Highest
   effort; defer until proven necessary.

---

## Verification (all phases)
Serve Klein NVFP4 on `max-serve`; generate a fixed prompt+seed image via
`/v1/responses`; compare to the bf16 baseline (structure) and, if built, the
Phase B host-dequant oracle (only quant error should differ). Phase C adds a
VRAM-residency check and a per-denoise-step latency number vs bf16. The
Apple/AMD W4A16 tests (`test/gpu/linalg/test_apple_fp4_matmul.mojo`,
`max/tests/integration/nn/test_linear_nvfp4_apple_gpu.py`) are the templates for
a CUDA W4A16 kernel test.

## Risks / sequencing
1. Phase A (scp) — lowest risk; unblocks loading with the clean upstream
   mechanism. Watch: `--model-override` may need Klein-executor plumbing.
2. Phase C (1 kernel build + wheel bump) — the main task, but **de-risked**: the
   dequant kernel is portable and already written+tested; there are two sibling
   launchers (Apple, AMD) to copy; no new PTX. Watch: the NVIDIA dense-GEMM entry
   point signature, and the July-3 wheel/kernel ABI lockstep on deploy.
3. Phase D — optional perf; only if materialize→dense is DRAM-walled at deep K
   (the upstream benchmark shows this happens at K≈18432). Native mma.sync is the
   only genuinely hard, multi-week item and is now fully optional.

## Deployment/box state (resume anchor)
- `max-serve` = `root@max-serve` (Proxmox LXC), GPU RTX PRO 6000 Blackwell
  **sm_120**, venv `/opt/max-serve/.venv` (py3.11), systemd `max-serve`, health
  `GET /health`. Currently serving bf16 Klein with dynamic batching.
- Deployed wheel is the **July-2** nightly; bump to **July-3 (`dev2026070306`)**
  before/at Phase C to match the rebased source for kernel builds.
- Combined NVFP4 repo `/mnt/models/klein-9b-nvfp4-full` (21 GB) becomes
  unnecessary once Phase A's `--model-override` path works (base repo + override
  to the NVFP4 safetensors).
- `.py` = live scp patch, no build. Kernel = `bazelw build` + artifact deploy +
  ABI-matched wheel.

# Plan: Native NVFP4 matmul for FLUX.2-Klein on sm_120 (RTX PRO 6000 Blackwell)

> Status: Proposed. Target = native sm_120 FP4 tensor-core matmul (full speed + VRAM), reached via de-risking phases.
> Mirror of the plan-mode file `~/.claude/plans/vast-booping-feather.md`.

## Context

`black-forest-labs/FLUX.2-klein-9b-nvfp4` is downloaded on `max-serve` but cannot run: the NVFP4 (`float4_e2m1fnx2`) block-scaled matmul in MAX is **SM100 (B200)-only**, and this box is **sm_120** (workstation Blackwell, compute cap 12.0). Loading is already solved (a combined bf16-base + NVFP4-transformer repo + forcing text_encoder/VAE to bf16); the remaining wall is compute:

```
_matmul_float4 -> dynamic_block_scaled_matmul:
ValueError: Both a_scales and b_scales must be rank 5 tensors
```

**Root cause (verified):** `quantize_dynamic_block_scaled` (`max/python/max/nn/kernels.py:6485`) gates on `_is_sm10x_gpu()` (`kernels.py:6403`, `startswith("sm_10")`). SM100 gets rank-5 "SF-atom" scales for the tcgen05/UMMA kernel; sm_120 falls into the rank-2 "CDNA4 proxy" branch, which `dynamic_block_scaled_matmul` (`kernels.py:6164`) rejects. Even un-gated, the Mojo kernel `block_scaled_matmul` (`max/kernels/src/linalg/fp4_quantization.mojo:1711`) hard-asserts B200, and **sm_120 has no `tcgen05`/UMMA** (`mojo/stdlib/std/sys/info.mojo:581`, `_SM_120X_ARCHS` at `:554`). The block-scaled FP4 MMA intrinsic (`UMMAKind.KIND_MXF4NVF4`) exists **only** as SM100 UMMA (`mojo/stdlib/std/gpu/compute/arch/mma_nvidia_sm100.mojo:58`); the general `mma.sync` layer (`mma_nvidia.mojo`) has **zero** FP4/block-scaled support.

**Goal (chosen):** a **native sm_120 NVFP4 block-scaled `mma.sync` matmul** — real FP4 tensor-core speedup + the ~13 GB VRAM win — reached through de-risking phases so we always have a correct, working fallback and a correctness oracle before the hard PTX bring-up. FLUX.2-Klein attention/MLP use plain `Linear` (`flux2/layers/flux2_attention.py:794`), so the only hot path to fix is `_matmul_float4` (`max/python/max/nn/quant_ops.py:73`); the fused-QKV float4 path (`quant_ops.py:455`) is an LLM path FLUX.2 never hits.

## Scope note on the build boundary
- `.py` changes deploy by **scp + restart** (no build). Every Mojo kernel change requires **`./bazelw build //max/kernels/...`** and deploying rebuilt kernel artifacts into the vendor wheel's `_interpreter_ops/__mojocache__/` (heavy, ABI-coupled). Phase 2 is the first time we cross this line and stand up the kernel build/deploy loop; Phase 3 lives entirely on the Mojo side.
- All changes are gated behind `not _is_sm10x_gpu()` / an explicit sm_120 predicate. bf16 layers never enter `_matmul_float4`, so the working bf16 pipeline is untouched.

---

## Phase 0 — Loading (already done; needs committing)
Deployed on `max-serve`, uncommitted in the fork: `flux2_klein_executor.py` (`self._component_encoding`), `flux2/components/vae_decoder.py` + `image_encoder.py` (force VAE bf16). Plus the combined repo `/mnt/models/klein-9b-nvfp4-full` and a `bazelw`-free CLI recipe (`--model-path <combined> --quantization-encoding float4_e2m1fnx2`). Action: commit these as "NVFP4 loading support (SM100-ready)".

## Phase 1 — Host-dequant validation (Python only, NO build) — correctness oracle
Prove the FLUX.2 NVFP4 math end-to-end with zero Mojo work; produce the reference image every later phase is checked against.
- In `max/python/max/pipelines/architectures/flux2/nvfp4_weight_adapter.py` (`convert_nvfp4_state_dict`, ~`:146`), port the numpy of `_dequantize_nvfp4_to_bf16` (`max/tests/integration/nn/ep/test_ep_moe_fp4.py:58-94`): per nvfp4 layer, combine `weight` (uint8 `[out,in//2]`), `weight_scale` (f8e4m3 `[out,in//16]`, already row-major after `_deinterleave_scales` at `nvfp4_weight_adapter.py:97`), `weight_scale_2` (f32 scalar) → one bf16 `[out,in]` `.weight`; drop the FP4 tensors. Nibble order already correct via `_swap_fp4_nibbles` (`:57`).
- Make those layers plain bf16 `Linear`: gate on sm_120 (or env flag) to make `nvfp4_layers_bfl` empty (`flux2/components/denoise_compute.py:214`) and pass `quant_config=None`, taking the `x @ weight.T` path (`max/python/max/nn/linear.py:576`).
- **Deploy:** scp `.py` only. **Verify:** load combined repo, generate fixed prompt+seed, save as the NVFP4 reference image. (No VRAM win — weights are bf16 — but the pipeline is proven.)

## Phase 2 — GPU dequant kernel + fused dequant→bf16 GEMM (first bazelw build) — working NVFP4 with VRAM win
Stand up the kernel build/deploy loop and get NVFP4 running on-device at ~bf16 speed with weights kept packed.
- **New Mojo kernel** `dequant_nvfp4` (new `max/kernels/src/linalg/nvfp4_dequant.mojo`, modeled on `mxfp4_dequant.mojo:48-121`): `SF_VECTOR_SIZE=16` (`fp4_utils.mojo:28`), scales `float8_e4m3fn` (`fp4_utils.mojo:32`, plain f32 cast — drop E8M0 handling), extra `* weight_scale_2`. Reuse `cast_uint_to_fp4e2m1` (`fp4_utils.mojo:98`) unchanged. Register op `mo.dequant.nvfp4` in `max/kernels/src/graph_compiler/builtin_kernels/quantization.mojo` (mirror `Struct_dequant_mxfp4` at `:777`).
- **Python:** add `nvfp4_dequant(...)` wrapper to `kernels.py` (mirror `mxfp4_dequant` at `:6337`); in `_matmul_float4` (`quant_ops.py:73`), before `:96`, branch `if not _is_sm10x_gpu(): return x @ nvfp4_dequant(weight, weight_scale, weight_scale_2).T` (activations stay bf16; `input_scale` unused).
- **VRAM caveat:** a standalone dequant of a constant weight may be constant-folded (bf16 in the artifact — no VRAM win). To guarantee the win, promote to a **fused dequant→GEMM** single kernel modeled on `mxfp4_matmul_sm90.mojo:27-95` (dequant in smem/regs → the sm_120 `multistage_gemm` bf16/fp8 GEMM at `matmul/gpu/__init__.mojo:651`). This fused kernel is the structural bridge into Phase 3.
- **Deploy:** `bazelw` kernel build + artifact deploy, then scp the `.py`. **Verify:** generate, pixel-compare to the Phase 1 reference; confirm no rank-5 / no B200 assert.

## Phase 3 — Native sm_120 FP4 block-scaled `mma.sync` matmul (the goal)
Real FP4 tensor-core throughput. This is the hard, PTX-level, multi-week work.
1. **New PTX intrinsics (foundational, does not exist today):** add sm_120 block-scaled FP4 `mma.sync` (the `mma.sync.aligned.*.kind::mxf4nvf4.block_scale` family, valid on `ptx87`/`sm_120a` per `mojo/stdlib/std/gpu/host/info.mojo:1047`) to the Mojo GPU layer alongside `mma_nvidia.mojo` (which has none) — the sm_120 analogue of the SM100 UMMA `KIND_MXF4NVF4` (`mma_nvidia_sm100.mojo:58`). Includes the register scale-operand plumbing.
2. **sm_120 scale layout:** the SM100 rank-5 SF-atom interleave (`kernels.py:6487`) is a tcgen05 artifact; sm_120's warp-level `mma.sync` wants a different scale register layout. Add an sm_120 branch to `quantize_dynamic_block_scaled` (`kernels.py:6485`) + a matching `block_scales_interleave` variant (`kernels.py:6695`), or use rank-2 directly if the sm_120 MMA accepts it.
3. **Kernel:** new `nvfp4_matmul_sm120.mojo` reusing `fp4_utils.mojo` (unpack) + the `matmul/gpu` tiling/TMA/scheduler infra, modeled structurally on the SM100 kernel (`grouped_matmul_sm100_1d1d.mojo:1148`) but on the `mma.sync` model (not UMMA/tcgen05). Dispatch: add an sm_120 branch in `block_scaled_matmul` (`fp4_quantization.mojo:1669`) before the B200 assert at `:1711`.
4. **Python dispatch:** replace `_is_sm10x_gpu()` with an `_is_fp4_tensorcore_gpu()` predicate matching sm_120/sm_121 in `quantize_dynamic_block_scaled` (`kernels.py:6485`), `dynamic_block_scaled_matmul` (`kernels.py:6164`), and `_matmul_float4` (`quant_ops.py:96`); route sm_120 to the native path instead of the Phase 2 dequant fallback.
- **Deploy:** `bazelw` kernel build + artifacts + scp `.py`. **Verify:** pixel-compare to the Phase 1 reference (correctness), then benchmark per-denoise-step latency vs Phase 2 dequant and the bf16 baseline to confirm the speedup. Keep the Phase 2 dequant path as the fallback for any non-sm_10x/non-sm_120 GPU.

---

## Critical files
- Python dispatch: `max/python/max/nn/quant_ops.py` (`_matmul_float4:73`), `max/python/max/nn/kernels.py` (`quantize_dynamic_block_scaled:6411`, `dynamic_block_scaled_matmul:6127`, `block_scales_interleave:6695`, `_is_sm10x_gpu:6403`).
- Mojo kernels: new `max/kernels/src/linalg/nvfp4_dequant.mojo` + `nvfp4_matmul_sm120.mojo`; `max/kernels/src/graph_compiler/builtin_kernels/quantization.mojo` (op registration); `fp4_quantization.mojo:1669-1711` (dispatch/gate); new sm_120 MMA intrinsics near `mojo/stdlib/std/gpu/compute/arch/mma_nvidia.mojo`.
- Reuse (unchanged): `fp4_utils.mojo` (`cast_uint_to_fp4e2m1:98`, `E2M1_TO_FLOAT32:39`); `mxfp4_dequant.mojo` + `mxfp4_matmul_sm90.mojo` (models); `grouped_matmul_sm100_1d1d.mojo:1148` (SM100 reference); `matmul/gpu` multistage GEMM (sm_120 bf16 path); `test_ep_moe_fp4.py:58-94` (correctness reference).
- Loading (Phase 0): `flux2_klein_executor.py`, `flux2/components/{vae_decoder,image_encoder}.py`, `flux2/nvfp4_weight_adapter.py`.

## Verification (all phases)
Load the combined NVFP4 repo on `max-serve`; generate a fixed prompt+seed image via `/v1/responses`; compare against the Phase 1 host-dequant reference (structure should match; only quant error differs). Phase 3 adds a per-denoise-step latency benchmark vs Phase 2 and bf16. Escalating oracle: Phase 1 image is the ground truth for Phases 2 and 3.

## Risks / sequencing
1. Phase 1 (scp) — lowest risk; unblocks + gives the oracle.
2. Phase 2 (1 kernel build) — establishes the build/deploy loop; delivers usable NVFP4 (VRAM win, ~bf16 speed). If the fused kernel is deferred, watch for constant-folding erasing the VRAM win.
3. Phase 3 — **the hard part**: sm_120 block-scaled FP4 `mma.sync` intrinsics **do not exist in Mojo today and must be written**; then a full kernel + scale layout + hardware perf bring-up. Highest risk/reward. Phases 1–2 guarantee a correct, shippable fallback the whole time.

---

## Deployment/box state at time of planning (for resume after compaction)
- `max-serve` = `root@max-serve` (Proxmox LXC), GPU RTX PRO 6000 Blackwell **sm_120**, venv `/opt/max-serve/.venv` (py3.11), systemd `max-serve`, health `GET /health`.
- Currently serving **bf16 FLUX.2-Klein-9B** with dynamic batching (`MODULAR_PIXEL_MAX_BATCH_SIZE`), healthy. NVFP4 override removed.
- Combined NVFP4 repo exists at `/mnt/models/klein-9b-nvfp4-full` (21 GB, real copies: base bf16 components + NVFP4 transformer as `transformer/diffusion_pytorch_model.safetensors`).
- Fork branches already pushed: `fix/flux2-vae-fused-conv-arch-gate`, `fix/zimage-turbo-load-and-output`, `fix/klein-num-images-batching`, `feat/klein-dynamic-batching`. The Phase-0 NVFP4 loading patches are deployed on the box but **uncommitted** in the fork.

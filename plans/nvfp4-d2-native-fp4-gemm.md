# Plan: D2 — native sm_120 NVFP4 (W4A4) FP4×FP4 GEMM

> Date: 2026-07-05. Status: **Proposed / starting.** Goal: the first NVFP4 kernel that
> actually **beats bf16** on sm_120, by running the native FP4 block-scaled tensor cores
> (FP4×FP4→f32) instead of dequantizing to bf16. Both risks retired: the **MMA works**
> (`nvfp4_mma_spike.mojo`, GO) and **W4A4 image quality holds** (`nvfp4-w4a4-accuracy-check.md`,
> all prompts passed). Execution: **main agent, step by step** (no subagents for the build —
> see memory `feedback-no-subagents-for-main-work`).

## The validated building block (from the spike — reuse verbatim)
`mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3`
(raw inline PTX, NOT the #3227-broken llvm intrinsic). Per-lane (group=lane>>2 0..7, tid=lane&3 0..3):
- **A** (16×64 e2m1): 4×u32 = 32 nibbles. a0: row=group, k[tid*8..+7]; a1: row=group+8, same k; a2: row=group, k[32+tid*8..+7]; a3: row=group+8, same. (Each u32 = 8 nibbles along K, nibble j at bit 4j.)
- **B** (64×8 e2m1, TN/transpose_b): 2×u32. col=group; b0: k[tid*8..+7]; b1: k[32+tid*8..+7].
- **SFA** (16×4 ue4m3): 1×u32 = 4 K-block scales; row owned by tid∈{0,1} (tid0→group, tid1→group+8).
- **SFB** (4×8 ue4m3): 1×u32 = 4 K-block scales for col=group; owned by tid==0.
- **C/D** (16×8 f32): 4×f32; d0→(group,2tid), d1→(group,2tid+1), d2→(group+8,2tid), d3→(group+8,2tid+1).
- Scale selectors are `{0,0}` for both (hardcoded in `MMA_ASM`).

## Design — symmetric FP4 GEMM (both operands fp4, hardware applies block scales)
W4A4 pipeline for `out[M,N] = A_bf16 @ W_fp4^T`:
1. **Quantize activation** bf16 `A[M,K]` → packed fp4 `[M,K//2]` + fp8-e4m3 block-16 scales `[M,K//16]` (per-tensor `input_scale` + per-block). Same packed format the weight already ships in — so A and B are **symmetric** into the GEMM.
2. **FP4×FP4 block-scaled GEMM**: tiled kernel using the spike atom; hardware applies SFA (act block scales) + SFB (weight block scales); f32 accumulate.
3. **Epilogue**: `out_bf16 = (f32_acc * input_scale * weight_scale_2)` — mirror the existing `dynamic_block_scaled_matmul` per-tensor fold (`tensor_sf = weight_scale_2 * input_scale`).

The GEMM is a standard tiled tensor-core GEMM; the only novelties vs a bf16 GEMM are (a) fp4
(4-bit) operands packed 2/byte, (b) the per-block scale operands, (c) the mma atom. Model the
tiling/smem/pipeline on `_multistage_gemm_gpu.mojo` (read-only reference); the fragment/scale
**loads** are new (fp4, so ldmatrix doesn't directly apply — start with manual per-lane loads
mirroring the spike's `build_fragments`, then optimize).

## Milestones (each: build + numeric-test on max-build, in the main agent)
**GEMM correctness is decoupled from the quantizer** — M1a–c feed HOST-quantized fp4 inputs
(like the spike) so the GEMM is validated in isolation; the sm_120 activation quantizer is M1d.

- **M1a — single m16n8k64 tile from SMEM.** Extend the spike: stage A(16×64), B(8×64), SFA,
  SFB into SMEM (packed fp4 + ue4m3), then have each lane load its a0-3/b0-1/sa/sb from SMEM
  (mirror `build_fragments` indexing on-device), run ONE mma, store. Verify vs the spike's
  `host_reference`. **Proves the on-device fragment+scale load path — the crux.**
- **M1b — K-loop.** Accumulate over K (K/64 strips) into c0-3 for one [16,8] output tile.
  Verify vs a full-K host ref (random fp4 inputs).
- **M1c — M/N tiling + epilogue.** Block tile [BM,BN] across warps; grid over M,N; per-tensor
  scale fold + bf16 store; M/N/K edge guards. Verify vs a W4A4 f32 host ref. New file
  `max/kernels/src/linalg/matmul/gpu/nvfp4_w4a4_cuda.mojo`, launcher
  `nvfp4_w4a4_matmul_cuda(c, a_fp4, a_scales, b_fp4, b_scales, ..., ctx)`.
- **M1d — sm_120 activation quantizer.** bf16 `[M,K]` → packed fp4 `[M,K//2]` + fp8-e4m3
  scales `[M,K//16]`, per-block-16 dynamic (amax/6 → fp8), per-tensor input_scale. Options:
  Mojo kernel (clean packing) or graph ops (reuse the `_snap_to_e2m1`/fake-quant math from
  `plans/tools/`). Numeric-check the packed output snaps to the e2m1 grid.
- **M1e — wire + op + python + test.** Op `mo.matmul.block.scaled.cuda.w4a4`; python helper +
  `_matmul_float4` W4A4 branch gated by `MODULAR_NVFP4_W4A4=1` (keep W4A16 default). End-to-end
  numeric test vs a W4A4 host reference (quantize A+W to fp4, dequant, f32 matmul) within tol.
- **M2 — benchmark** vs bf16 / D1-fused / materialize at M∈{64,128,256}, real shapes, L2-flushed.
  **Expect fused-native to finally BEAT bf16** (~2-4× fp4 tensor-core rate + 4-bit bandwidth).
- **M3 — end-to-end.** Route the pipeline to W4A4, render the 4 hard prompts, confirm quality
  matches the validated W4A4-sim (`nvfp4-w4a4-accuracy-check.md`), measure the render speedup,
  deploy.

## Oracle / tests
- M1a/b: the spike's `host_reference` (E2M1[A]·SFA · E2M1[B]·SFB summed over k), extended to full K.
- M1c–e: a W4A4 host reference (numpy/Mojo): quantize A and W to fp4+scales, dequantize, f32 matmul.
- The **D1 fused kernel + materialize kernel remain the W4A16 oracles**; W4A4 differs numerically
  (activation also quantized) so use the W4A4 ref + a tolerance, and the **image test** (M3) is
  the real quality gate (already pre-validated by the sim).

## Risks / notes
- **fp4 fragment loads:** ldmatrix is for 16-bit; fp4 packs 2/byte. Start with manual per-lane
  SMEM loads (correct, mirrors the spike), optimize (vectorized/ldmatrix-on-packed) in M2. This
  is the main unknown.
- **Scale SMEM layout → sa/sb registers:** must deliver the 4 ue4m3 K-block bytes to the owning
  lanes (tid∈{0,1} SFA, tid==0 SFB) with the `{0,0}` selector convention. Validated for one atom;
  extend carefully; the host_reference catches errors.
- **Small M is a GEMV-ish regime** (M=1 wastes the 16-row MMA) — same caveat as before; real M
  (image tokens ~64-512) is fine. Split-K may still help occupancy (reuse D1's lesson).
- **Keep D1 as the safe fallback** (shippable win over today's NVFP4); D2 is the upside.

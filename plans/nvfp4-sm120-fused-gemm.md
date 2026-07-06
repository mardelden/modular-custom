# Plan: Phase D-1 — Fused NVFP4 W4A16 GEMM for sm_120 (in-mainloop dequant)

> Status: **Proposed 2026-07-05**. Follows Phase C (materialize→dense, shipped).
> Branch `feat/nvfp4-sm120-native-kernel`. Motivated by a measured inference
> regression: NVFP4 is ~4× **slower** than bf16 at small images (128²), because
> the Phase C kernel re-materializes the whole weight to a bf16 `[N,K]` DRAM
> buffer on **every** forward pass (cost ∝ weight size, independent of image
> size → dominates when M/tokens are small). See "Why" below.

## Goal
Make NVFP4 **match or beat** bf16 at all image sizes while keeping the 4-bit VRAM
win, by dequantizing the packed FP4 weight **inside the GEMM mainloop** (per tile,
in shared memory) instead of a separate full-weight materialize pass. No new PTX
(reuse the existing bf16 `mma.sync`). This is Phase D option 1 from
`nvfp4-sm120-native-kernel.md`.

## Why (root cause, measured)
- Phase C `nvfp4_w4a16_matmul_cuda` per call: `enqueue_create_buffer[bf16](N*K)` +
  `enqueue_fp4_materialize` (writes `2·N·K` bf16 to DRAM) + `_matmul_gpu` (reads it
  back). Weight DRAM traffic ≈ **4.5·N·K** vs bf16's **2·N·K** (~2.25×), plus
  double the kernel launches. At 128² the GEMM is tiny, so this fixed dequant
  dominates → **0.66 s/img NVFP4 vs ~0.17 s/img bf16** (klein, live).
- A fused GEMM reads only the **0.5·N·K** packed FP4 (¼ of bf16's weight bytes)
  and never materializes bf16 → in the small-M **bandwidth-bound** regime it can be
  **faster than bf16**. Amortizes to ≈ bf16 at large M.

## Design (fork one kernel, change only the B producer)
Fork base (verified sm_120 bf16 path): on sm_120 `_matmul_gpu`
(`matmul/gpu/__init__.mojo:411`) skips the tcgen05 SM100 dispatch (gated on
`_has_blackwell_tcgen05()`, `:629`) and H100 (`:638`), falling through to
**`multistage_gemm_kernel`** (`matmul/gpu/_multistage_gemm_gpu.mojo`) — cp.async →
smem → `mma.sync`, Ampere configs.

The B (weight) DRAM→smem seam in that kernel:
- prologue prefetch: `_copy_tensor_to_sram[…swizzle_b](b_smem_tile, b_iter[])`
  (`_multistage_gemm_gpu.mojo:359–364`)
- mainloop prefetch: same call (`:482–499`)
- consumer (unchanged): `mma_op.load_b(b_warp_tile, …)` (`:451`) reads **bf16** from
  `b_smem_iter[]`.

**Change:** replace the two B-copy sites with a cooperative dequant stage that
writes the SAME bf16 `b_smem_tile`:
1. new kernel args `b_packed_iter` (`uint8 [N, K//2]`) + `b_scales_iter`
   (`float8_e4m3fn [N, K//16]`) replacing bf16 `b_iter`.
2. `_stage_dequant_b(b_smem_tile, b_packed_iter[], b_scales_iter[])` — threads
   cooperatively: extract nibble → `E2M1_TO_FLOAT32[nib] * abs(scale)` → store bf16.
   Reuse `fp4_utils.mojo` (`E2M1_TO_FLOAT32` LUT `:39`, `decode_e2m1_to_f32` `:107`)
   and Apple's `_stage_dequant` structure (`apple/fp4_matmul.mojo:351–431`).
3. advance packed/scale iters by `BK//2` / `BK//16` instead of `b_iter` by `BK`.
4. mma mainloop, A-load, pipelining, epilogue: **untouched** (smem stays bf16).

`weight_scale_2` stays a graph-level post-matmul scalar fold (as Phase C).

**Implementation shape (refined after reading the fork target):** `multistage_gemm`
is a 3-layer, heavily-parameterized kernel (`multistage_gemm` launcher →
`multistage_gemm_kernel` → `multistage_mma` inner loop, `_multistage_gemm_gpu.mojo:182`)
shared by **every** bf16/fp8 matmul (split-k, next-op fusion, many dtypes). Editing it
in place risks destabilizing the whole matmul path. So prefer a **self-contained new
kernel** in `nvfp4_w4a16_fused_cuda.mojo` modeled on Apple's purpose-built
`AppleM5Fp4MatMul` (already load-packed→decode→smem→mma shaped;
`apple/fp4_matmul.mojo:316–486`), swapping Apple's Metal simdgroup MMA for NVIDIA
`mma.sync` (`mma_nvidia.mojo` m16n8k16 bf16) and its threadgroup copy for `cp.async`
(idioms referenced from `multistage_mma`). Fallback if the port is too costly: add
fp4 **siblings** `multistage_mma_fp4` / `..._kernel_fp4` / `..._fp4` (copy-modify, do
NOT touch the shared originals). Decide at M1 start after reading Apple's kernel in
full; leaning self-contained-port for isolation.

## Op / Python wiring
- New launcher `nvfp4_w4a16_fused_matmul_cuda` in a new
  `matmul/gpu/nvfp4_w4a16_fused_cuda.mojo` (keep Phase C kernel as fallback/oracle).
- New op `mo.matmul.weight.only.block.scaled.cuda.fused` in
  `builtin_kernels/linalg.mojo` (mirror the Phase C op).
- Python `_cuda_weight_only_block_scaled_matmul_fused` in `kernels.py`; route the
  `_matmul_float4` CUDA branch (`quant_ops.py:137`) to it behind a switch
  (env `MODULAR_NVFP4_FUSED=1` during A/B; make default once validated).

## Milestones
- **M1 — correctness.** Fork kernel + dequant B-load + op + python. Build on
  max-build (`./bazelw build //max/kernels/... //max:builtin_kernels //max:linalg`).
  Numeric unit test vs the Phase C materialize kernel (same inputs → same output
  within bf16 tolerance). Extend `test_linear_nvfp4_cuda_gpu.py`. **Done = bitwise-
  close on sm_120.**
- **M2 — perf.** Micro-bench fused vs materialize vs bf16 at small M (128² regime)
  and large M. Tune BK depth (Apple: deeper BK amortizes decode). **Done = fused ≤
  bf16 latency at small M, ≈ bf16 at large M.**
- **M3 — end-to-end.** Route `_matmul_float4` to fused; render Klein NVFP4, pixel-
  compare to known-good `/tmp/klein_nvfp4_fixed.png`; measure 6×128² wall time vs
  bf16. **Done = image matches + speedup on the real test.**
- **M4 — ship.** Rebuild baked wheels, republish `/mnt/wheelhouse/max`, deploy.

## Decision
D1 (fused cooperative-SMEM dequant + reuse bf16 `mma.sync`) over D2 (native
`mma.sync kind::mxf4nvf4.block_scale`). See `decisions/0001-d1-fused-vs-d2-native.md`.
D2 deferred: sm_120 has no such intrinsic in Mojo today; multi-week new PTX; only
needed if M2 shows we're **compute**-bound (not bandwidth-bound) at large M.

## Dev loop & constraints
- Kernel builds/tests on **max-build** (sm_120); local macOS cannot build GPU Mojo.
  Edit → `bazelw build` (no GPU needed) → GPU numeric test (needs a free GPU).
- **Shared GPU:** klein is currently serving (≈86 GB used). All code+build work
  needs no GPU; batch the on-GPU numeric/bench tests when the GPU is free (or
  coordinate stopping klein). Do NOT stop klein without explicit OK.
- Correctness oracle = the Phase C materialize kernel (identical decode math).

## Risks
- **Decode not overlapped with MMA** (Apple note): scalar-ALU decode can't co-issue
  with tensor MMA → amortize over deeper BK. If decode-bound, M2 tunes BK/threads.
- **Register/smem pressure** from the extra staging + scale loads → may cap BK/BN.
- **Scale addressing** (block-16 along K, per-N-row) must match the packing the
  weight adapter produced (rank-2 `[N, K//16]`, deinterleaved) — same contract the
  Phase C kernel already validates.

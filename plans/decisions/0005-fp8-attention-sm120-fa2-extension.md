# 0005 — Full-fp8 (e4m3) attention on sm_120 by extending the generic FA2 kernel

Status: Accepted (K2 kernel numerically validated 2026-07-06)
Context: [[nvfp4-fused-d1-d2-status]] (W4A4 GEMM done) → D3 profiling showed the Klein
render is **attention-bound** (denoise 987 ms = 81%, identical bf16/W4A4, 99% util →
dominated by the bf16 `mo.mha.no_cache` flash attention). The only NVFP4-specific render
win left is fp8 attention. Phase Q (in-graph e4m3 fake-quant simulation) showed quality is
acceptable and that quantizing the softmax probabilities P is essentially free
(`full` ≈ `qkv`), so the target is the *full* fp8 path: fp8 Q·K^T **and** fp8 P·V.

## Decision

Do NOT write a standalone flash-attention kernel, and do NOT thread fp8 through the whole
FA-dispatch. Instead reach the existing **non-pipelined `mha_single_batch` FA2 kernel** with
an e4m3 config and add the ×256 P-lift. Rationale below (alternatives rejected).

sm_120 fp8 config (all comptime-gated on `dtype.is_float8()`, bf16 path value-identical):
- `BK = 64` (fp8 MMA is m16n8k32 → K=32; BK=64 keeps `num_k_mmas = BK/MMA_K` even, which
  `multistage_mma` requires).
- `num_pipeline_stages = 2` (forced for fp8 — see Lesson 2; BK=64 with the default 4 stages
  is *numerically wrong*).
- `WN = BN` → `num_warps_n = 1` (the SINGLE-warp geometry bf16 production uses at depth
  128). fp8 must NOT use `WN=BN/2` (multi-warp): that path is untested at depth 128 and is
  numerically WRONG even for bf16 (Lesson 5). Instead fp8 stays single-warp but routes P
  through the smem-staging path (`shared_mem_bytes` reserves p_smem for fp8 at
  num_warps_n==1, and the P-stage gate becomes `num_warps_n > 1 or q_type.is_float8()`),
  because the single-warp register-reuse P·V only casts f32→bf16.
- `is_shared_kv = False` for fp8 (the pipelined kernel's vectorized copy can't do f32→e4m3;
  fp8 tiles are half-size so the non-pipelined kernel fits sm_120's ~100 KB smem).
- ×256 P-lift in `_copy_frag_to_smem_nvidia` (clears e4m3's subnormal floor, mirrors the
  datacenter sm100 kernel) + matching ÷256 in the `mha_single_batch` epilogue.
- e4m3 mma.sync m16n8k32 works through the **standard** TensorCore/`multistage_mma`
  abstraction on sm_120a — no raw inline PTX (unlike the fp4 block-scaled case). Proven by
  `fp8_mma_spike.mojo`.

Files: `nn/attention/mha_utils.mojo` (`MHAConfig` BK/stages/WN + `shared_mem_bytes` +
`q_smem_size`), `nn/attention/gpu/mha.mojo` (`is_shared_kv` gate, epilogue ÷256 + output
store width), `nn/attention/mha_utils.mojo` `_copy_frag_to_smem_nvidia` (P-lift),
`layout/tensor_core.mojo` (`_load_b_nvidia` fp8 fragment fix). Test:
`test/gpu/nn/mha_fp8_e4m3_test.mojo` (host ref + bf16 controls).

## Alternatives rejected

- **Standalone flash kernel** — full rewrite of tiling + online softmax; high bug surface.
- **Thread fp8 through the shared FA dispatch** — ~6 fp8 gates all hardcode
  "fp8 ⟹ sm100/AMD"; fragile to relax without breaking those paths. The raw-tensor rank-4
  overload (the one `mo.mha.no_cache` uses) has **no** dtype-reject assert, so e4m3 already
  flows to the generic FA2 launch — extending the *kernel* is enough.

## Lessons (the four stacked bugs, and the method)

Each bug hid the next; the first two produce NO assert / NO crash.

1. **smem accounting counts in dtype units but `warp_scratch` is f32.** For sub-4-byte
   dtypes `MHAConfig.shared_mem_bytes` under-allocates the scratch (fp8 = 4×) →
   `CUDA_ERROR_ILLEGAL_ADDRESS` with no assert (LayoutTensor asserts validate the *layout*,
   not the *allocation*). Fix: add `(4/size_of[dtype] − 1)·warp_scratch_smem_size()`.
2. **The kernel is only correct when BK×stages ≈ depth.** BK=64 with stages=4 gives WRONG
   NUMBERS even in bf16 (no crash). Caught by the controls matrix. fp8 (BK=64) forces
   stages=2.
3. **A real typo in dead-until-now code.** `tensor_core.mojo` `_load_b_nvidia`
   non-transposed fp8 branch read `frags[0,1]` from a `(2,1)` fragment (OOB); correct is
   `frags[1,0]` (PTX m16n8k32 B: b0 = k rows `4·(lane%4)..+3`, b1 = +16). First path to
   exercise it finds the bug.
4. **Output store must vectorize by the OUTPUT dtype width** (bf16→8), not the input
   (fp8→16). Coincidentally equal for bf16-in/bf16-out, so latent.
5. **The multi-warp geometry (WN<BN) is broken at depth 128** — untested in production
   (bf16 always uses WN=BN there). The isolated fp8 test PASSED with narrow inputs but the
   RENDER was garbage (MAE ~55, PSNR ~10 dB); wide-range inputs then failed the isolated
   test (max_rel 466, wrong at d≥64 = the 2nd warp's depth-half), and a bf16 control at the
   same WN=64 geometry ALSO failed — proving it's a geometry bug, not fp8. Fix: single-warp
   + smem P staging. Meta-lesson: **test kernels with WIDE-RANGE inputs** (small structured
   inputs → near-uniform softmax → magnitude-dependent bugs hide under the abs-tolerance
   gate), and **a passing isolated test at one shape/input does not imply the render works**.

**Debug method (compute-sanitizer CANNOT attach to Mojo binaries — verified with both the
CUDA 12.9 and 13.0 sanitizer builds; "terminated before first instrumented API call"):**
- bazel mojo tests run `-D ASSERT=all` → device asserts print exact `file:line` + thread =
  the sanitizer substitute — but only *after* fixing allocation-level overruns (Lesson 1),
  which asserts can't see.
- **Controls matrix:** run bf16 through the exact new geometry one axis at a time
  (WN, then BK, then stages) vs the default bf16 config. Separates "geometry bug
  (dtype-independent)" from "dtype bug" in ~2 builds. The bf16 BK64/stages4 control failing
  *numerically* was the pivotal clue for Lesson 2.
- Comptime stage-skip probes (`_FP8_PROBE` = skip QK / P-stage / PV) localize an assert to a
  pipeline stage in ~90 s/run.

## Status / next

K2 done: `mha_fp8_e4m3_test` PASS vs exact host ref; bf16 regression clean (the one failing
bf16 sink test reproduces on the pristine tree — pre-existing on sm_120). Remaining:
K3 (op registration + `MODULAR_NVFP4_FP8_ATTN=1` routing in `flux2_attention.py` +
render-gate vs Phase-Q `full` sim), K4 (fp8-vs-bf16 attention bench + serve A/B; target
denoise < 987 ms, render < bf16 ~1.5 s).

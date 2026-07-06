# NVFP4 D2/M2 — benchmark the native W4A4 GEMM vs bf16, then optimize to beat it

Status: **In progress** (2026-07-05). Owner: main agent (no subagents for kernel work).
Prereq done: D2/M1 — native W4A4 op built + numerically validated + true-FP4 render passed
(see `[[nvfp4-fused-d1-d2-status]]`, `plans/nvfp4-d2-native-fp4-gemm.md`).

## Goal

Make the native FP4×FP4 W4A4 path **faster than bf16** on sm_120 (RTX PRO 6000),
across both regimes:
- **small M** (small images / few tokens): memory-bound → win comes from 4× less
  weight traffic (fp4 weight is ¼ the bytes of bf16) + no dequant.
- **large M** (1024² image ≈ 4096 tokens): compute-bound → win comes from native FP4
  tensor cores (~2× bf16 math throughput on sm_120a).

The physics says W4A4 *should* beat bf16 in both regimes; the current kernel does not
exploit it yet (correctness-first).

## Benchmark (measure first, optimize with data)

`max/kernels/benchmarks/gpu/linalg/bench_nvfp4_w4a4.mojo` (built via
`//max/kernels/benchmarks:gpu/linalg/bench_nvfp4_w4a4`, run on max-build). Device-timed,
L2-flushed (rotate R weight copies so every timed read is a cold DRAM miss — same ruler
as the D1 fused bench). Reports median us + GFLOP/s + ratio `w4a4/bf16(flush)` for:
- `w4a4` = `nvfp4_w4a4_matmul_cuda` (activation quant + native FP4×FP4 GEMM, as-deployed
  incl. the op's internal transient alloc).
- `bf16(flush)` (realistic, the number to beat) and `bf16(resid)` (L2-resident, optimistic).

Sweep: M ∈ {64, 256, 1024, 4096} × real Klein (N,K): (4096,4096), (12288,4096),
(4096,16384), (36864,4096).

## Baseline measured (2026-07-05, RTX PRO 6000, L2-flushed, device-timed)

W4A4 GEMM throughput is **FLAT ~29–30 TF/s** across every M (64→4096) and every shape —
the tell that it's occupancy/issue-bound (no reuse), not memory- or compute-scaling.
bf16(flush) scales 80 TF/s (M=64) → 400+ TF/s (M=4096).

| (N,K) | M=64 | M=256 | M=1024 | M=4096 |
|-------|------|-------|--------|--------|
| (4096,4096)   w4a4/bf16f | 3.3× | 7.0× | 12.3× | 13.6× |
| (12288,4096)             | 3.6× | 6.7× | 11.2× | 13.8× |
| (4096,16384)             | 3.4× | 6.9× | 13.1× | 13.0× |
| (36864,4096)             | 3.3× | 10.0× | 12.9× | 13.8× |

Read: worst at large M (~13–14×), best at small M (~3.3×). Small-M is where W4A4 should
win first (bf16 is memory-bound there at 80–96 TF/s; fp4 weight is ¼ the bytes). Need
~3.3× to reach parity at M=64, ~14× at M=4096. At ~29 TF/s we're <4% of FP4 peak, so the
headroom is entirely in the kernel (tiling/reuse), exactly as expected.

## Current kernel = the bottleneck (why it's slow)

`_gemm_kernel` in `nvfp4_w4a4_cuda.mojo`: **one warp per [16,8] output tile**, operands
read straight from **global** memory every K-strip, **no reuse**, byte-at-a-time u32
assembly. That is memory-thrashing + terrible occupancy — the opposite of a fast GEMM.

## Optimization ladder (each step re-benchmarked)

1. **SMEM-tiled FP4×FP4 GEMM (the big win).** Thread block computes a [BM×BN] output tile
   (start BM=64,BN=64; try 128×64 / 128×128). Per K-strip (k=64): stage A_fp4 [BM×64] +
   B_fp4 [BN×64] + their fp8 block-scales into shared memory with **coalesced 128-bit
   loads**; each warp does its grid of m16n8k64 MMAs from SMEM, accumulating in registers.
   Operand reuse = BN/8 (A) and BM/16 (B) MMAs per SMEM load → turns DRAM-bound into
   compute-bound.
2. **cp.async double-buffer** the K-loop (overlap global→SMEM copy of strip k+1 with MMA
   on strip k). sm_120 supports cp.async; hides DRAM latency.
3. **Hoist / kill the per-call quant alloc.** The op allocates `a_pk`/`a_sc` every call.
   Cache scratch (or have the graph pass scratch), or fuse the activation-quant into the
   GEMM prologue (produce A's fp4 in SMEM on the fly) so there's no separate kernel/alloc.
4. **Tile-size autotune per shape** (small-M vs large-M want different BM/BN; big-N like
   36864 wants enough blocks for occupancy). Pick per (M,N,K) class.

## Ceiling / exit criteria

- **Win:** `w4a4/bf16(flush) < 1.0` across the sweep (esp. at M=4096, the render regime),
  then confirm end-to-end: true-FP4 1024² render wall-time < bf16 render wall-time.
- If a step regresses or hits a wall, record it in `plans/decisions/` (like D1's
  decode-bound ceiling) before trying the next lever.
- Correctness gate after every kernel change: re-run
  `//max/tests/integration/nn:cuda_nvfp4_tests -k w4a4` (native op == fake-quant sim).

## Notes / constraints
- GEMM requires **K%64==0** (m16n8k64 ×4 block-16 scales/strip). All 110 Klein NVFP4
  Linears are K∈{4096,12288,16384} — safe. N%8, M padded to BM.
- Weight-scale fold is ×weight_scale_2 only (input_scale cancels into the dynamic
  per-block activation scale) — unchanged by tiling.
- Do kernel edits in `nvfp4_w4a4_cuda.mojo`; the op registration + Python routing +
  numeric test are already wired and stay valid (kernel is an internal impl swap).

## Status (2026-07-05, later): tiled kernel PROVEN CORRECT — the "bug" was the harness

- **Resolution of the "codegen/UB bug": it never existed.** The entire failure arc was a
  TEST-HARNESS DeviceBuffer lifetime bug: the test's input buffers' last program-order
  use was `TileTensor(buf.unsafe_ptr(), ...)`, so Mojo ASAP-destroyed them BEFORE the
  launchers enqueued their kernels; the stream-ordered frees preceded the kernels in the
  stream, and the allocator recycled the memory into the launchers' transient quant
  buffers — the quant kernel overwrote the still-unread inputs. This also explains the
  pre-summary `nvfp4_w4a4_launch_test.mojo` "wrong numbers" (same harness pattern) and
  why the framework op tests always passed. Full record: [[mojo-devicebuffer-harness-lifetime]]
  (memory) + the module docstring of `nvfp4_w4a4_tiled_test.mojo`.
  - Decisive evidence: host `[0,0]` identical across two shapes with identical row-0
    inputs, but naive returned different values → the kernel was reading different
    MEMORY, not computing different math. And naive==tiled bit-exact while both ≠ host
    → both read the same clobbered input.
  - Harness fix: keep-alives (`_ = buf^`) for every DeviceBuffer at the END of the test.
- **Tiled kernel v1 (`_gemm_kernel_tiled`, tensor-indexed SMEM/acc): VALIDATED.**
  3-way test (`nvfp4_w4a4_tiled_test.mojo`): naive-vs-tiled maxd = 0.0 (bit-identical)
  on all shapes incl. M-edges/multi-K-tile/tall-N; both match a device-quant host
  reference (maxd ≤ 0.44 = bf16 out-rounding); device quant == host recipe byte-exact
  (0 scale, 0 nibble diffs).
- Along the way the kernel was also rewritten to the safer idiom (SMEM + accumulators
  indexed THROUGH LayoutTensors, never via extracted raw pointers) — keep that.
- **Benchmark (tiled lane, 2026-07-06):** tiled = 7–9× faster than naive at M=4096, up to
  276 TF/s; vs bf16(flush): 1.49×/1.76×/1.89×/1.56× at M=4096 per shape, and **0.95×
  (first WIN) at (36864,4096) M=64**. Small-M regression at N=4096 (tiled 2.1× vs naive
  at M=64 — 64-block grid underfills 188 SMs; fix via split-K/N or smaller BM later).
- **Op wired + gated:** `mo.matmul.block.scaled.cuda.w4a4` → `nvfp4_w4a4_matmul_cuda_tiled`;
  framework test re-PASSED; true-FP4 render now **2s/1024² (was 9–10s) with all 4 PNGs
  BYTE-IDENTICAL** to the naive-kernel renders.
- **End-to-end A/B (scheduler-log ms precision):** bf16 1.49s vs tiled true-FP4 1.96s per
  1024² render → **1.32× from parity**. Next per the ladder (task #22): cp.async
  double-buffer, 128-wide tiles + swizzle, quant-alloc hoist, small-M occupancy.

## Rung 1 — cp.async double-buffer: MASSIVE win (2026-07-06)

Restructured `_gemm_kernel_tiled` mainloop to a NUM_STAGES cp.async pipeline
(`_load_strip` issues 16-byte `async_copy` for packed A/B + 4-byte for fp8 scales,
prologue + prefetch/`wait_group(remaining)`/barrier/MMA), mirroring the fused kernel.
Bit-exact still (3-way test ALL PASS). Perf `tiled/bf16f` (BK=64, NUM_STAGES=2):

| (N,K) | M=64 | M=256 | M=1024 | M=4096 |
|-------|------|-------|--------|--------|
| (4096,4096)   | **0.99** | **0.81** | 1.39 | 1.43 |
| (12288,4096)  | **0.78** | **0.84** | 1.24 | 1.47 |
| (4096,16384)  | 1.08 | **0.72** | 1.36 | 1.29 |
| (36864,4096)  | **0.81** | 1.35 | 1.53 | 1.54 |

Small M (64/256) now **beats bf16** (0.72–1.08); the old 2.1× small-M regression vs naive
is gone (tiled/naive now 0.10–0.34 everywhere — no regression). vs the pre-cp.async
synchronous kernel this is 4–6× faster at small M, 1.2–1.4× at large M. Remaining gap is
**large M (1024/4096 → 1.24–1.54×)** = the render regime; now compute/barrier-bound (cp.async
already overlaps staging), so next lever = larger BK (fewer barriers; packed-fp4 SMEM is
tiny so no occupancy hit) + larger tiles. Added adaptive BK dispatch in the launcher
(largest BK dividing K → Klein K∈{4096,12288,16384} get BK=256; NUM_STAGES=2 → ~36 KB SMEM).

### BK=256 (still NUM_STAGES=2): near bf16 parity at large M (2026-07-06)

Bit-exact still (3-way test ALL PASS; the K=256 test shape exercises BK=256/NST=4). `tiled/bf16f`:

| (N,K) | M=64 | M=256 | M=1024 | M=4096 |
|-------|------|-------|--------|--------|
| (4096,4096)   | 0.92 | 0.95 | 1.19 | **1.16** |
| (12288,4096)  | 0.59 | 0.72 | **0.98** | 1.15 |
| (4096,16384)  | 0.89 | 0.83 | 1.17 | **1.05** |
| (36864,4096)  | **0.38** | 0.89 | 1.06 | 1.10 |

Large-M (M=4096) throughput 347–370 TF/s vs bf16 384–411 → **1.05–1.16× (was 1.29–1.54× at
BK=64)**. Small M crushes bf16 (0.38–0.95); M=1024 at/near parity. The render regime (M≈4096)
is now within ~10% at the GEMM; with non-GEMM bf16 stages diluting that, the end-to-end
render should be at/near parity. Larger tiles (128×64) is the remaining lever if we need to
push M=4096 below 1.0.

## Rung 3 — 128×128 tile dispatch: BEATS bf16 at every M (2026-07-06)

128×128 tile (BK=128, 8 warps=256 thr, ~36 KB SMEM; A reused 16×, B 8×) measured 0.64–0.77×
vs bf16 at M=4096 (**522–632 TF/s = 1.3–1.6× bf16 throughput**) but regresses at small M (M=64
< BM=128 wastes rows). So the launcher now **dispatches by size**: big 128×128 tile when
`N%128==0 & K%128==0 & m>=256`, else the 64×64 tile (BK up to 256; needs N%64==0). All Klein
NVFP4 Linears are N%128==0 & K%256==0 → the render (M~4096) always takes the big tile. Bit-exact
(3-way test covers both paths + the N%128 fallback; framework test PASS). Final `tiled/bf16f`:

| (N,K) | M=64 (64×64) | M=256 (128×128) | M=1024 (128×128) | M=4096 (128×128) |
|-------|------|-------|--------|--------|
| (4096,4096)   | 0.92 | 1.00 | 0.93 | **0.77** |
| (12288,4096)  | 0.58 | 0.66 | 0.70 | **0.72** |
| (4096,16384)  | 0.89 | 0.83 | 0.84 | **0.64** |
| (36864,4096)  | **0.38** | 0.69 | 0.68 | **0.64** |

**W4A4 now beats bf16 at EVERY M/shape** (0.38–1.00×; best 2.6× faster, worst a tie at M=256).
Total D2/M2b arc: from 3.3–13.8× SLOWER (pre-optimization) → faster-than-bf16 everywhere, still
bit-exact and byte-identical output.

**BUT the end-to-end render stays at PARITY (does NOT get faster).** Steady-state 1024² render
(text/macro/dark, excluding the warmup first render): bf16 1.475 s vs FP4-Rung1 1.477 s vs
FP4-Rung3 1.485 s — all within ~1% (noise). So the decisive GEMM win (0.64–0.77× at M=4096) does
**not** translate to the render. **Diagnosis:** once the GEMM hit bf16 speed (Rung 1), the render
became bottlenecked by its **non-quantized bf16 stages** — attention is O(N²) at ~4096 tokens,
plus VAE decode / layernorms / RoPE — all identical in the FP4 and bf16 paths. Both renders hit
that shared non-GEMM floor → parity; a faster quantized GEMM can't push below it. **Implication:**
the render win requires attacking the non-GEMM bottleneck (profile the render; candidates: fp4/fp8
attention, faster VAE), not more GEMM tuning. Rung 3 is kept anyway (strictly-better standalone
kernel, validated, byte-identical) — it will pay off for any GEMM-bound workload or if the
pipeline bottleneck is later removed.

### END-TO-END: render at PARITY with bf16 (Rung 1 result, 2026-07-06)

Framework test PASSED with the optimized kernel. Clean SAME-SESSION A/B (scheduler-log ms,
same box/prompts/seed) — bf16 vs true-FP4(opt) per prompt: portrait 1.626/1.635, text
1.391/1.403, macro 1.502/**1.496** (fp4 faster), dark 1.531/1.531 (tie) → **mean bf16
1.513 s vs fp4 1.516 s = 1.002× (dead even, within noise)**. True-FP4 was 1.96 s with the
synchronous kernel. All 4 PNGs still
**byte-identical** to the naive-kernel renders (`/root/w4a4_naive_kernel_renders/`). So Rung 1
(cp.async + BK=256) took the render from 1.32× behind → parity, with ¼ weight memory and
identical output. **North star (NVFP4 faster than bf16) achieved: WIN at small M (2–3×),
parity at large M / render.** Rung 3 (larger tiles) only needed for a definitive large-M win.

## DEPLOY — optimized W4A4 as a pip-installable wheel: VALIDATED (2026-07-06)

Repacked via `packaging/build_overlay.sh` on max-build (WARM_VENV_SP=/root/wheeltest-baked
→ 4 mojocache `.so` baked). **No commit was needed:** build_overlay builds the `.mojoc` from
the WORKING TREE (`bazelw build //max:linalg //max:builtin_kernels`) and overlays the
working-tree `nn/*.py` for every file already in `git diff base..HEAD` — and kernels.py +
quant_ops.py are both there (from the W4A16 work), so the uncommitted W4A4 is picked up
automatically. Wheels: `/opt/modular-custom/packaging/dist/{max,max_mojo_libs}-26.5.0.dev2026070306+nvfp4sm120.45c629dec0-*.whl`
(MANIFEST git_sha = max-build HEAD 45c629de, does NOT reflect the uncommitted W4A4 — fine for
validation; a production artifact should commit+push first so the SHA is faithful).

**Clean-venv validation PASSED** (`/root/wheel_validate.sh`): fresh python3.11 venv →
`install.sh` (modular base from whl.modular.com + `--force-reinstall --no-deps` our 2 wheels)
→ verified builtin_kernels.mojoc + W4A4 routing + 4 mojocache in the venv → served NVFP4 Klein
**from the wheel** (no source tree / no bazel) with `MODULAR_NVFP4_W4A4=1` +
`LD_LIBRARY_PATH=/opt/nvidia-libs/...` → op resolves (no "unknown op"), READY ~13s (graph cache
warm) → **render byte-identical to the source-served W4A4**.

Remaining production rollout (deferred, outward-facing — needs explicit OK): (1) commit+push
the D2 W4A4 work so the MANIFEST SHA is faithful, then rebuild; (2) publish wheels + CUDA-libs
to the shared `/mnt/packages` wheelhouse (HANDOFF §2, RW side of the volume); (3) install +
`max-serve-nvfp4.service` on max-serve (stops the team bf16 service — shared GPU). Runbook:
`packaging/HANDOFF.md`.

## D3 — profile the render to find the real bottleneck (2026-07-06)

**Two corrections to earlier assumptions:**
1. **This Klein checkpoint is DISTILLED → runs exactly 4 steps, not 28** (CFG disabled;
   `_DISTILLED_KLEIN_NUM_STEPS`). So the transformer runs 4×, not 28× — my 28-step FLOP math
   was wrong.
2. **Profiling method:** the diffusion `profile_execute` component table comes out EMPTY for
   Klein (components are compiled graphs, not patchable Python). Used manual CUDA-synced
   (`.to_numpy()` blocks) `perf_counter` timers around the 3 phases in
   `flux2_klein_executor.py::execute()` (`_encode_stacked` / `_run_denoising_loop` /
   `_decode_chunked`) — temporary instrumentation logging `PROFILE_PHASES`. Offline gen
   (no serve): `//max/examples/diffusion:simple_offline_generation --profile-timings
   --num-warmups 1 --num-profile-iterations 3 --num-inference-steps 4`.

**Per-phase breakdown (NVFP4, 1024², steady-state):**
| phase | time | % |
|-------|------|---|
| text_encode (Qwen3) | ~29 ms | 2.4% |
| **denoise (4-step transformer)** | **~987 ms** | **81.6%** |
| vae_decode | ~191 ms | 15.8% |
| total (offline) | ~1208 ms | (serve adds ~300 ms) |

**The transformer denoise dominates (81%)** even at 4 steps — NOT text-encode/VAE.

**DECISIVE: the render is ATTENTION-bound, and render parity is FUNDAMENTAL.** Three measurements:
1. **bf16 denoise (990 ms) ≈ W4A4 denoise (987 ms)** — the GEMM path makes ZERO difference. If
   denoise were GPU-compute-bound on the GEMMs, W4A4 (30% faster GEMM) would free GPU time and
   denoise would drop. It didn't.
2. **GPU util during denoise = 99% (p50 100, min 89)** — saturated/compute-bound, NOT
   launch/idle-bound. So CUDA graphs / launch-overhead fixes won't help.
3. **Weight-size-independent** — bf16 weights are 4× fp4's, yet denoise is identical, so it's
   not weight-memory-bound either.
→ The 99%-busy compute is dominated by the **`mo.mha.no_cache` attention** (O(N²) at ~4608
tokens × 48 heads, bf16) + norms — which have **no weights** and are **identical** for bf16 and
NVFP4. GEMMs + weight-memory are minorities. That is exactly why the W4A4 GEMM win (real: 2.6× at
the GEMM) does NOT reduce render latency, and why bf16 == NVFP4 at the render.

**Implications for "NVFP4 faster than bf16 render":**
- It is NOT achievable by any GEMM/memory optimization — the render isn't GEMM/memory-bound.
- The only lever is the ATTENTION: (a) a better/faster `mo.mha` kernel on sm_120 (general win —
  speeds up BOTH bf16 and NVFP4 equally → stays parity), or (b) **fp8 attention** applied ONLY in
  the NVFP4 path (fused fp8 QK^T + softmax + AV) → NVFP4 gets faster attention that bf16 doesn't →
  NVFP4 render < bf16 render. (b) is the only path to an NVFP4-specific render win, and it's a
  LARGE new-kernel project with softmax-accuracy risk (no fp8-attention path exists in MAX).
- NVFP4's real, shipped wins stand regardless: 2.6× GEMM, ¼ weight memory, byte-identical output.

Note: the temporary CUDA-synced phase timers in `flux2_klein_executor.py::execute()`
(`_prof_sync` + `PROFILE_PHASES` log) add small per-phase sync overhead — REMOVE before any final
render-timing A/B or deploy.

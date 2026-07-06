# ADR 0001 — sm_120 NVFP4 speed: D1 fused cooperative-SMEM dequant, not D2 native FP4 mma.sync

Date: 2026-07-05
Status: Accepted
Context: Phase C (materialize→dense) is correct + saves VRAM but is ~4× slower than
bf16 at small images (re-materializes the whole weight to a bf16 DRAM buffer every
forward pass). We want quantized to be as fast or faster than bf16.

## Decision
Implement **D1**: a fused GEMM that dequantizes packed FP4 weight tiles to bf16 in
shared memory **inside the mainloop**, then feeds the **existing bf16 `mma.sync`**
tensor-core path. Fork `multistage_gemm_kernel` (the kernel sm_120 already uses for
bf16) and change only the B (weight) DRAM→smem load. No new PTX.

## Alternatives considered
- **D2 — native block-scaled FP4 `mma.sync` (`kind::mxf4nvf4.block_scale`).**
  Real FP4 tensor-core throughput. Rejected *for now*: the only block-scaled FP4
  MMA in the tree is `KIND_MXF4NVF4`, which is **tcgen05/UMMA, sm_100a-only**
  (`mma_nvidia_sm100.mojo:58`; `tcgen05.mojo:18`). sm_120 has no tcgen05. A warp-
  level `mma.sync…mxf4nvf4` for sm_120a exists in PTX (ptx 8.7) but has **no Mojo
  intrinsic today** → multi-week new-PTX work. Defer until proven compute-bound.
- **Keep Phase C materialize.** Rejected: correct but the small-M regression stands;
  extra ~2.25× weight DRAM traffic + double launches is inherent to the approach.
- **Cache the dequantized bf16 weight once.** Rejected: keeping bf16 resident throws
  away the 4-bit VRAM win — that is just running bf16.

## Consequences
- Captures the DRAM-bandwidth win (read ¼ the weight bytes) → faster at small M,
  ≈ bf16 at large M, VRAM win preserved.
- Compute at large M is still bf16-MMA-bound (no 2× FP4 tensor-core speedup) — that
  upside needs D2. Revisit only if M2 shows a large-M compute wall.
- Low risk: forks a battle-tested kernel; correctness oracle = the Phase C kernel.

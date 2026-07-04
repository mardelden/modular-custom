# NVFP4 sm_120 — build machine requirements & deploy runbook

> Audience: deployment/infra team preparing a dedicated Modular **build
> container**. Goal: compile our fork (July-3 nightly + a new CUDA NVFP4 kernel)
> into an artifact that runs on the **max-serve** box (NVIDIA **sm_120**, RTX PRO
> 6000 Blackwell) so FLUX.2-Klein can run its transformer in 4-bit NVFP4.
>
> Branch to build: **`feat/nvfp4-sm120-native-kernel`** on the fork
> (`git@github.com:mardelden/modular-custom.git`).

## 1. What this branch changes (what needs compiling)

One **new Mojo GPU kernel** + registration, plus two Python files (no build):

- `max/kernels/src/linalg/matmul/gpu/nvfp4_w4a16_cuda.mojo` *(new)* — the kernel.
- `max/kernels/src/graph_compiler/builtin_kernels/linalg.mojo` — registers the
  op `mo.matmul.weight.only.block.scaled.cuda`. **This is the part that must be
  compiled into the kernel package.**
- `max/python/max/nn/kernels.py`, `max/python/max/nn/quant_ops.py` — pure Python
  dispatch; live-patchable, no compile.

The kernel dequantizes NVFP4 weights to bf16 and runs the existing dense bf16
GEMM, so it depends only on in-tree kernels (`_matmul_gpu`,
`enqueue_fp4_materialize`). No new external deps, no PTX, no compiler change.

## 2. Build machine requirements

| Requirement | Value / notes |
|---|---|
| OS | Linux **x86_64** (match max-serve's CPU arch + glibc; Ubuntu 22.04/24.04 fine). |
| GPU | An **NVIDIA sm_120** (Blackwell, compute cap 12.0) GPU — ideally the *same* model as max-serve (RTX PRO 6000). Lets the box build **and** run the kernel test + a real render. A GPU is not strictly required to *compile*, but is required to run the validation targets; strongly recommended to have it. |
| CUDA | A CUDA toolkit that targets **compute_120 / sm_120a** → **CUDA 12.8+** (Blackwell). Note: `./bazelw` fetches the Mojo GPU toolchain itself via `@mojo_gpu_toolchains`; a host CUDA driver compatible with sm_120 (recent NVIDIA driver, 570+) must be present for GPU test/run. |
| Toolchain | Repo's **`./bazelw`** wrapper (bootstraps Bazel + the pinned Mojo/MAX toolchains). Nothing to install manually beyond a JDK-less bazelw prereq set (git, python3, a C/C++ toolchain, `unzip`, `zip`). |
| Pinned nightly | `bazel/mojo.MODULE.bazel` pins **MAX 26.5.0.dev2026070306 / Mojo 1.0.0b3.dev2026070306** (July-3), from `https://whl.modular.com/nightly`. The build fetches these automatically — the box needs network egress to `whl.modular.com` and the Bazel registries. |
| Resources | Bazel build of the kernels is heavy: **≥16 cores, ≥32–64 GB RAM, ≥100 GB free disk**. First build with a cold cache can take a long time. |
| Network | Egress to `whl.modular.com`, GitHub, and Bazel Central Registry. **If your org has a BuildBuddy/remote-cache endpoint, wire it in** — it turns a multi-hour first build into minutes (see risks). |

## 3. Build steps (on the build box)

```bash
git clone git@github.com:mardelden/modular-custom.git
cd modular-custom
git checkout feat/nvfp4-sm120-native-kernel

# Sanity: the kernels compile (targets sm_120 via the GPU toolchain).
./bazelw build //max/kernels/...

# Confirm our new op registration compiles in the builtin kernel package:
./bazelw build //max:builtin_kernels

# (If a GPU is present) run the FP4 kernel tests as a smoke check:
./bazelw test //max/kernels/test/gpu/linalg/...
```

Then produce the **deployable artifact** (pick one model in §4).

## 4. Deployment model — pick one

**(A) Preferred — build & install the full MAX from our fork.** Build the MAX
install/wheel from this tree (July-3 + our changes) and install it on max-serve.
Everything is ABI-consistent (kernels + runtime + Python from one source), so
there is no surgical patching and no separate wheel-version bump.
- Investigate the exact packaging target with the team: `./bazelw run //:install`
  produces a dev install (compiler + runtime + **prebuilt kernel packages**);
  confirm it bundles our rebuilt `builtin_kernels`. If a distributable **wheel**
  target exists, prefer that for a clean container install.
- Deploy: install that artifact into the max-serve container's venv
  (`/opt/max-serve/.venv`).

**(B) Alternative — patch the existing vendor wheel.** Keep max-serve on the
July-3 vendor wheel and drop in just our rebuilt kernel package + scp the 2
Python files. Lighter, but you must ABI-match the July-3 wheel **exactly** and
locate the wheel's compiled kernel package to replace. More fragile than (A).

> Recommendation: **(A)**. One source, one ABI, fewer moving parts.

## 5. Runtime (max-serve) requirements

- Must run the **same July-3 nightly** the build used (`dev2026070306`) — model
  (A) guarantees this; model (B) requires bumping the wheel to July-3 first.
- Serve NVFP4 Klein with per-component overrides (base repo bf16, transformer
  NVFP4), using the upstream mixed-precision mechanism:
  ```bash
  MAX_SERVE_API_TYPES='["openai","responses"]' \
  max serve --model-path <klein-base-repo> \
    --model-override 'transformer.quantization_encoding=float4_e2m1fnx2' \
    --model-override 'transformer.weight_path=["<abs-path-to-klein-nvfp4.safetensors>"]' \
    --device-memory-utilization 0.9
  ```
  (This replaces the 21 GB combined-repo hack. Klein-executor override plumbing =
  our Phase A, still to be finished on the Python side.)

## 6. Validation (once deployed)

1. Server starts, `/health` OK, logs show the transformer loaded as
   `float4_e2m1fnx2` and no "unknown op `mo.matmul.weight.only.block.scaled.cuda`"
   error (that error = the kernel package was not rebuilt/installed).
2. Render a fixed prompt+seed via `/v1/responses`; image is coherent (not grey).
3. Confirm the **VRAM drop**: transformer resident set is ~4-bit-sized vs the
   bf16 baseline (the win); the dense bf16 weight is only a per-op transient.
4. Compare against the bf16 baseline image (structure matches; only quant error
   differs).

## 7. Key risks to verify first (biggest → smallest)

1. **Does `./bazelw build //max/kernels/...` complete on your box without
   Modular-internal infra?** The monorepo normally leans on BuildBuddy remote
   cache/execution. Verify a clean local build succeeds (and how long it takes)
   before committing to the container design. If you have a remote cache, wire it
   in via `.bazelrc`/`--config`.
2. **Which target yields the installable artifact that includes our rebuilt
   kernels** (model A). Confirm `//:install` (or a wheel target) bundles the
   freshly built `builtin_kernels`, not a prebuilt/cached one.
3. **sm_120 targeting** — confirm the GPU toolchain compiles for compute_120
   (may need a `--config` or `SUPPORTED_GPUS` selection). A physical sm_120 GPU
   on the build box removes all ambiguity.
4. **glibc/arch match** between build box and max-serve.

## 8. Interim option (no build) — Phase B

If you want NVFP4-Klein producing correct images on sm_120 **before** the build
container is ready: we can land a **host-dequant** path (dequant the NVFP4
weights to bf16 in NumPy at load; layers become plain bf16 `Linear`). This is
**Python-only (scp, no build)**, correct output, but **no VRAM win** (weights sit
in memory as bf16). It also produces the reference image to validate the kernel
against. Say the word and we'll ship it while the build box is prepared.

## 9. Handoff checklist

- [ ] Build container: Linux x86_64, sm_120 GPU, CUDA 12.8+, ≥16c/≥32GB/≥100GB,
      egress to `whl.modular.com` + GitHub + Bazel registries.
- [ ] `git checkout feat/nvfp4-sm120-native-kernel` from the fork.
- [ ] `./bazelw build //max/kernels/...` succeeds (report time / whether remote
      cache is needed).
- [ ] Decide artifact model (A preferred) and produce the install/wheel.
- [ ] max-serve on July-3 nightly; deploy artifact; run §6 validation.

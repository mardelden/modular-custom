# Deploy-team runbook — custom MAX (NVFP4 / sm_120) wheels

Build, publish, and install our custom MAX as pip-installable **repacked wheels**.
Background/why in `README.md`; this is the operational runbook.

## What the artifact is
Two wheels that overlay our changes onto the stock vendor MAX:
- `max-<ver>+nvfp4sm120.<sha>-cp311-…manylinux_2_34_x86_64.whl` — Python delta
  (NVFP4 W4A4 dispatch + fused ws2, VAE unfused attention, PNG-encode speedup,
  fp8-attention routing (gated off), Klein pipeline). Per-interpreter (cp311).
- `max_mojo_libs-<ver>+nvfp4sm120.<sha>-py3-none-any.whl` — our compiled Mojo
  kernels (`builtin_kernels`, `linalg`, `layout`, `builtin_primitives`).

The closed vendor binaries (compiler, `_core.so`, `libmax.so`) are untouched and
still come from `whl.modular.com`. The `+nvfp4sm120.<sha>` local tag pins the
exact build and lets multiple versions coexist.

## Current build (already available)
`max-build:/opt/modular-custom/packaging/dist/` — base `26.5.0.dev2026070306`,
git `76b48d0827` (branch `feat/nvfp4-fp8-attention`), **kernel_caches_baked: 4**.
Grab it: `scp -r root@max-build:/opt/modular-custom/packaging/dist/ .`
Perf on this build: FLUX.2-Klein 1024² serve render **855 ms** (NVFP4, MODULAR_NVFP4_W4A4=1)
vs **1501 ms** full-bf16 = 1.76× at ¼ weight memory; clean-venv validated byte-identical.

## 1. Build a new set (build box with GPU + `./bazelw`, e.g. max-build)
```bash
cd /opt/modular-custom
git fetch origin && git checkout feat/nvfp4-fp8-attention
git reset --hard origin/feat/nvfp4-fp8-attention          # MUST be at HEAD:
                                                          # a lagging HEAD drops
                                                          # .py files from the diff
# Point WARM_VENV_SP at a venv that has ALREADY served the model, so the
# precompiled kernel caches (max/**/__mojocache__/*.so) get baked in — otherwise
# fresh installs cold-JIT the framework ops on first serve (a multi-minute hang).
WARM_VENV_SP=/root/wheeltest-w4a4-clean/lib/python3.11/site-packages \
  packaging/build_overlay.sh                              # -> packaging/dist/
# (WARM_VENV_SP = a venv that has ALREADY installed THIS wheel and served once.
#  The 76b48d0827 build used /root/wheeltest-w4a4-clean, warmed by
#  /root/wheel_validate.sh. A stale-SHA warm venv bakes the WRONG kernel caches.)
```
Rebuild when: the branch changes, OR the pinned nightly bumps
(`MAX_PACKAGE_VERSION` in `bazel/mojo.MODULE.bazel`). ABI lockstep: the `.mojoc`
must be built against the same nightly the vendor wheels are — `build_overlay.sh`
reads that version automatically.

**Kernel caches (`__mojocache__`) — two-pass build.** No vendor wheel ships the
JIT'd framework-op `.so`, so first serve on a cold box compiles them (`mojo build
… --emit shared-lib`, minutes/op → effective hang). `build_overlay.sh` bakes them
in by harvesting from `WARM_VENV_SP`. Flow for a fresh nightly:
1. Build once (no warm venv yet — prints a "no __mojocache__ harvested" WARNING).
2. `install.sh <venv> packaging/dist` and serve the model once (warms
   `<venv>/…/site-packages/max/**/__mojocache__/*.so` — 4 files).
3. Re-run `build_overlay.sh` with `WARM_VENV_SP=<that venv site-packages>` — now
   the wheel bakes the 4 `.so`. The `.so` are hash-named + deterministic, so they
   only need re-harvesting on a nightly bump.
Verify: `unzip -l dist/max-*.whl | grep -c __mojocache__` should be **4**, and
`MANIFEST.json` `kernel_caches_baked: 4`.

## 2. Publish to the shared wheelhouse (from the RW side of `nvme-vg-shared`)
`/mnt/packages` is read-only inside the LXC containers, so copy from the Proxmox
host (or wherever the volume is mounted rw):
```bash
mkdir -p /mnt/packages/max-wheels
cp /path/to/dist/*.whl /path/to/dist/MANIFEST.json /mnt/packages/max-wheels/
# one-time: stage the CUDA runtime so every container gets cuBLAS the same way
#   /mnt/packages/cuda-libs/  <- libcublas.so.12, libcudnn.so.*, libcusparse..., etc.
```
Keep old versions; the `+<sha>` tag keeps them distinct.

## 3. Install into a container venv
```bash
V=26.5.0.dev2026070306; TAG=nvfp4sm120.76b48d0827
pip install --find-links /mnt/packages/max-wheels \
    "modular==$V" "max==$V+$TAG" "max_mojo_libs==$V+$TAG"
```
`modular==$V` pulls the closed base wheels + Python deps from `whl.modular.com`;
`--find-links` makes pip prefer our repacked `max`/`max_mojo_libs`.
(Equivalently: `packaging/install.sh <venv> /mnt/packages/max-wheels`.)

**CUDA runtime:** `pip install modular` does NOT pull cuBLAS/cuDNN. The box must
provide them — either they're already in the venv, or set
`LD_LIBRARY_PATH=/mnt/packages/cuda-libs/...` (see the serve unit).

## 4. Serve
Fill placeholders in `max-serve-nvfp4.service`, install it, `systemctl enable
--now max-serve-nvfp4`. Render via `POST /v1/responses`.

## 5. Verify
Fixed prompt+seed render should match the known-good clean image (red vase +
yellow tulips). Failure modes:
- **"unknown op … block.scaled.cuda"** → the `.mojoc` weren't picked up (check
  `modular/lib/mojo/` has our `builtin_kernels.mojoc`).
- **washed-out image** → the VAE fix `.py` didn't land (rebuild at branch HEAD).
- **"symbol not found: cublasCreate_v2"** → no CUDA runtime on the loader path.

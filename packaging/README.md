# Custom MAX packaging (NVFP4 / sm_120)

Ship this fork's custom MAX (NVFP4 W4A16 sm_120 kernel + VAE arch-gate fix +
Klein changes) as **repacked pip wheels** — a normal `pip install`, no source
tree or bazel on the serving box.

## Why repack, not build from source

A `modular` wheel can't be built from this fork: the Mojo **compiler**,
**`_core.so`** (MLIR/graph-compiler bindings) and **`libmax.so`** + runtime libs
are closed, download-only binaries. The parts we changed are open and
rebuildable — Mojo kernels (compiled to `.mojoc` *with* the downloaded compiler)
and Python. So we bake those two into the stock vendor wheels:

| Our change | Rebuilds | Repacked into |
| --- | --- | --- |
| `nvfp4_w4a16_cuda.mojo`, op registration in `builtin_kernels/linalg.mojo` | `linalg.mojoc`, `builtin_kernels.mojoc` (+ deps `layout`, `builtin_primitives` the vendor ships baked-in, not standalone) | `max_mojo_libs` → `modular/lib/mojo/` |
| `quant_ops.py`, `kernels.py`, `vae_flux2.py`, Klein executor/components, … | — (pure Python) | `max` |

The graph compiler resolves kernels by scanning `modular/lib/mojo/*.mojoc`, so
adding our `builtin_kernels.mojoc` there makes our new op resolve — verified on a
clean `pip install` (the op registers, the model serves).

## Build (on a build box with `./bazelw`, e.g. max-build)

```bash
packaging/build_overlay.sh [OUT_DIR=packaging/dist] [BASE_COMMIT=<nightly base>]
```

Produces in `OUT_DIR`: `max-<V>+nvfp4sm120.<sha>-cp311-…whl`,
`max_mojo_libs-<V>+nvfp4sm120.<sha>-…whl`, and `MANIFEST.json`. The `+local` tag
keeps `pip install modular==<V>` satisfied while preventing a stock wheel from
silently replacing ours.

> **ABI lockstep:** the `.mojoc` must be built from the same nightly the vendor
> wheels are (`MAX_PACKAGE_VERSION` in `bazel/mojo.MODULE.bazel`). Ensure the
> checkout is at the branch HEAD before building so the Python diff is complete.

## Install (on the serving box, into its venv)

```bash
packaging/install.sh /opt/max-serve/.venv packaging/dist
```

This bumps the base to the pinned nightly and force-reinstalls our two wheels.
**CUDA runtime:** `pip install modular` does *not* pull cuBLAS/cuDNN — the box
must provide them (max-serve already has them in its venv). On a bare box:
`pip install nvidia-cublas-cu12==12.8.4.1 nvidia-cudnn-cu12 nvidia-cusparse-cu12
nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-nvjitlink-cu12`
and put their `lib` dirs on `LD_LIBRARY_PATH`.

## Serve

Fill `max-serve-nvfp4.service` placeholders (`@VENV@`, `@MODEL_BASE@`,
`@NVFP4_SAFETENSORS@`, `@CUDA_LIB_DIRS@`), install it, `systemctl enable --now
max-serve-nvfp4`. Render via `POST /v1/responses` (not `/v1/models`, which 500s
for diffusion pipelines).

## Verify

Compare a fixed prompt+seed render against the known-good `klein_nvfp4_fixed.png`
(a clean red-vase-with-yellow-tulips). A washed-out image means the VAE arch-gate
fix (`vae_flux2.py`) didn't land — check the `.py` overlay. An "unknown op" means
our `.mojoc` weren't picked up — check `modular/lib/mojo/`.

## Maintenance

A nightly bump = re-run `build_overlay.sh` after rebasing onto the new nightly.
`MANIFEST.json` records the base version + git sha for traceability.

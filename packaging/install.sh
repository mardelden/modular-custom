#!/usr/bin/env bash
# Install the custom repacked MAX (NVFP4/sm_120) into a target venv.
# Usage: install.sh <venv-path> <dist-dir>
#   <dist-dir> holds the repacked wheels + MANIFEST.json from build_overlay.sh.
set -euo pipefail
VENV="${1:?usage: install.sh <venv> <dist-dir>}"
DIST="${2:?usage: install.sh <venv> <dist-dir>}"
PIP="$VENV/bin/pip"
V="$(python3 -c "import json;print(json.load(open('$DIST/MANIFEST.json'))['base_max_version'])")"

echo "== ensuring pinned base modular==$V (nvidia CUDA libs are NOT deps and are preserved) =="
"$PIP" install "modular==$V" --extra-index-url https://whl.modular.com/nightly/simple/

echo "== force-reinstalling our repacked wheels (no-deps so pip keeps our versions) =="
"$PIP" install --force-reinstall --no-deps \
  "$DIST"/max-*+nvfp4sm120.*.whl \
  "$DIST"/max_mojo_libs-*+nvfp4sm120.*.whl

echo "== CUDA runtime check (MAX matmul needs cuBLAS/cuDNN; NOT a pip dep of modular) =="
if ls "$VENV"/lib/python*/site-packages/nvidia/cublas/lib/libcublas.so* >/dev/null 2>&1; then
  echo "  cuBLAS present in venv."
else
  cat <<'MSG'
  WARNING: no cuBLAS in this venv. MAX inference will abort with
  "symbol not found: cublasCreate_v2". Provide the CUDA runtime, e.g.:
    pip install nvidia-cublas-cu12==12.8.4.1 nvidia-cudnn-cu12 \
      nvidia-cusparse-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
      nvidia-cusolver-cu12 nvidia-nvjitlink-cu12
  and set LD_LIBRARY_PATH to their lib dirs in the serve unit (see README).
MSG
fi
echo "== installed custom MAX $V into $VENV =="
"$VENV/bin/python" - <<'PY'
import max
print("  max ok:", getattr(max, "__version__", "n/a"))
PY

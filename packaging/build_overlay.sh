#!/usr/bin/env bash
# Build custom repacked MAX wheels from this fork.
#
# We can't build `modular` from source (compiler/_core/libmax are closed
# binaries), so we bake our two rebuildable parts into the stock vendor wheels:
#   - our compiled Mojo kernels (.mojoc)  -> repacked into `max_mojo_libs`
#   - our Python delta                    -> repacked into `max`
# Output: two `+nvfp4sm120.<sha>` wheels + a MANIFEST, installable with pip.
#
# Run on a build box with `./bazelw` (e.g. max-build). Usage:
#   packaging/build_overlay.sh [OUT_DIR] [BASE_COMMIT]
set -euo pipefail

REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO"
OUT="${1:-$REPO/packaging/dist}"
BASE_COMMIT="${2:-d35568099c}"   # July-3 nightly commit the vendor wheels track
PY_TAG="cp311"                    # target interpreter (max-serve = python 3.11)
PLATFORM="manylinux_2_34_x86_64"

V="$(grep -oE 'MAX_PACKAGE_VERSION = "[^"]+"' bazel/mojo.MODULE.bazel | cut -d'"' -f2)"
SHA="$(git rev-parse --short HEAD)"
TAG="nvfp4sm120.$SHA"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT"
echo "== base MAX version: $V   git: $SHA   tag: $TAG =="

# 1. Build the kernel packages we overlay (builtin_kernels transitively builds
#    its deps; we collect the four the vendor lacks/changes).
echo "== building kernels (./bazelw build //max:builtin_kernels //max:linalg) =="
./bazelw build //max:builtin_kernels //max:linalg
BB="bazel-bin/max/kernels/src"
declare -A MOJOC=(
  [builtin_kernels]="$BB/graph_compiler/builtin_kernels/builtin_kernels.mojoc"
  [builtin_primitives]="$BB/graph_compiler/builtin_primitives/builtin_primitives.mojoc"
  [layout]="$BB/layout/layout.mojoc"
  [linalg]="$BB/linalg/linalg.mojoc"
)

# 2. Download the two vendor wheels to repack.
BASEURL="https://whl.modular.com/nightly"
MAX_WHL="$WORK/max-$V-$PY_TAG-$PY_TAG-$PLATFORM.whl"
MML_WHL="$WORK/max_mojo_libs-$V-py3-none-any.whl"
echo "== downloading vendor wheels =="
curl -fsSL -o "$MAX_WHL" "$BASEURL/max/max-$V-$PY_TAG-$PY_TAG-$PLATFORM.whl"
curl -fsSL -o "$MML_WHL" "$BASEURL/max-mojo-libs/max_mojo_libs-$V-py3-none-any.whl"

# 3. Repack max_mojo_libs with our .mojoc (installed to modular/lib/mojo/).
MML_ARGS=()
for name in "${!MOJOC[@]}"; do
  MML_ARGS+=(--overlay "max_mojo_libs-$V.data/platlib/modular/lib/mojo/$name.mojoc=${MOJOC[$name]}")
done
echo "== repacking max_mojo_libs (${#MOJOC[@]} kernels) =="
python3 packaging/repack_wheel.py --wheel "$MML_WHL" --out "$OUT" --tag "$TAG" "${MML_ARGS[@]}"

# 4. Repack max with our Python delta (in-wheel path = strip `max/python/`).
mapfile -t PYFILES < <(git diff --name-only "$BASE_COMMIT"..HEAD -- max/python/max | grep '\.py$' || true)
MAX_ARGS=()
for f in "${PYFILES[@]}"; do
  MAX_ARGS+=(--overlay "${f#max/python/}=$REPO/$f")
done

# 4b. Bake precompiled Mojo kernel caches (max/**/__mojocache__/*.so). No vendor
# wheel ships these — MAX JIT-compiles the framework ops (_core_mojo,
# _kv_cache_ops, _distributed_ops) on first serve, which is a multi-minute cold
# hang. Harvest them from a WARM reference venv (one that has already served the
# model) and bake them in so fresh installs start warm. Deterministic
# (hash-named) + tiny (~1.4 MB), so portable across boxes.
WARM_VENV_SP="${WARM_VENV_SP:-/root/wheeltest/lib/python3.11/site-packages}"
NCACHE=0
if [ -d "$WARM_VENV_SP/max" ]; then
  while IFS= read -r so; do
    MAX_ARGS+=(--overlay "${so#"$WARM_VENV_SP"/}=$so")
    NCACHE=$((NCACHE + 1))
  done < <(find "$WARM_VENV_SP/max" -path '*__mojocache__*' -name '*.so' 2>/dev/null)
fi
if [ "$NCACHE" -eq 0 ]; then
  echo "WARNING: no __mojocache__/*.so harvested (set WARM_VENV_SP to a venv that"
  echo "         already served the model). Fresh installs will cold-JIT on first serve."
fi

echo "== repacking max (${#PYFILES[@]} python files + $NCACHE kernel caches) =="
python3 packaging/repack_wheel.py --wheel "$MAX_WHL" --out "$OUT" --tag "$TAG" "${MAX_ARGS[@]}"

# 5. Manifest (ABI-lockstep + provenance).
cat > "$OUT/MANIFEST.json" <<JSON
{
  "base_max_version": "$V",
  "git_sha": "$SHA",
  "local_tag": "$TAG",
  "py_tag": "$PY_TAG",
  "platform": "$PLATFORM",
  "kernels": [$(printf '"%s",' "${!MOJOC[@]}" | sed 's/,$//')],
  "kernel_caches_baked": $NCACHE,
  "python_files": ${#PYFILES[@]}
}
JSON
echo "== done. artifacts in $OUT: =="
ls -1 "$OUT"

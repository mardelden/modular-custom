#!/usr/bin/env bash
# K3: render the 4 Klein prompts with the REAL fp8 attention kernel
# (MODULAR_NVFP4_W4A4=1 MODULAR_NVFP4_FP8_ATTN=1). First serve triggers a bazelw
# Mojo rebuild (kernel changed) + graph compile, so allow a long ready wait.
# Output: /root/fp8real/<name>.png. Compare vs the Phase-Q `full` sim
# (/root/fp8sim/full_*.png) and the W4A4 baseline (/root/w4a4_naive_kernel_renders/).
set -uo pipefail

OUT=/root/fp8real
mkdir -p "$OUT"
STATUS="$OUT/run.status"
SERVE_LOG="$OUT/serve.log"
PY=/root/wheeltest-baked/bin/python
: > "$STATUS"; : > "$SERVE_LOG"

P_portrait='a close-up portrait photo of an elderly woman, deeply wrinkled skin, silver hair, sharp detail, soft studio light'
P_text='a vintage neon storefront sign that reads "OPEN 24 HOURS" glowing at night, reflections on wet pavement'
P_macro='a macro photo of a dragonfly resting on a dew-covered spider web, intricate detail, shallow depth of field'
P_dark='a dimly lit gothic cathedral interior lit only by candlelight, deep shadows, volumetric light rays'

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }
log "REAL fp8 attention render pass start"

pkill -f "pipelines" 2>/dev/null || true
pkill -f "serve_klein_nvfp4" 2>/dev/null || true
sleep 4
log "launching serve (MODULAR_NVFP4_W4A4=1 MODULAR_NVFP4_FP8_ATTN=1)"
setsid env MODULAR_NVFP4_W4A4=1 MODULAR_NVFP4_FP8_ATTN=1 \
  bash /root/serve_klein_nvfp4.sh >>"$SERVE_LOG" 2>&1 </dev/null &
CPID=$!
sleep 1
PGID=$(ps -o pgid= -p "$CPID" 2>/dev/null | tr -d ' '); [ -z "$PGID" ] && PGID="$CPID"
echo "$PGID" > "$OUT/pgid"

READY=0
for i in $(seq 1 2400); do
  if grep -q "Server ready" "$SERVE_LOG" 2>/dev/null; then READY=1; log "READY after ~${i}s"; break; fi
  if grep -qiE "symbol not found|Worker crashed|Traceback \(most|ABORT:|OUT_OF_MEMORY|unknown op|no operation|CUDA_ERROR|ILLEGAL|failed to allocate|Assert Error" "$SERVE_LOG" 2>/dev/null; then
    log "SERVE ERROR at ${i}s:"; tail -30 "$SERVE_LOG" | tee -a "$STATUS"; log "ALLDONE rc=11"; exit 11
  fi
  sleep 1
done
[ "$READY" = 1 ] || { log "NOT READY in 2400s"; tail -30 "$SERVE_LOG" | tee -a "$STATUS"; log "ALLDONE rc=10"; exit 10; }
sleep 3

render(){
  local name="$1" prompt="$2"; local out="$OUT/${name}.png"; rm -f "$out"
  local t0 t1; t0=$(date +%s)
  if "$PY" /root/render_klein.py "$out" "$prompt" >>"$STATUS" 2>&1; then
    t1=$(date +%s); log "render $name OK size=$(stat -c %s "$out" 2>/dev/null||echo 0) took=$((t1-t0))s"
  else
    local rc=$?; t1=$(date +%s); log "render $name FAILED rc=$rc took=$((t1-t0))s"
  fi
}
render portrait "$P_portrait"
render text     "$P_text"
render macro    "$P_macro"
render dark     "$P_dark"

log "teardown pgid=$PGID"
kill -TERM -- -"$PGID" 2>/dev/null || true
sleep 3
pkill -f "pipelines" 2>/dev/null || true
log "ALLDONE rc=0"

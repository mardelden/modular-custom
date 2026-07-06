#!/usr/bin/env bash
# Phase Q: render the 4 Klein prompts under fp8-attention SIMULATION modes and
# save PNGs for quality comparison vs the known-good W4A4 renders.
#
# For each MODE in the args (default: qkv full), serves Klein NVFP4 from source
# with MODULAR_NVFP4_W4A4=1 MODULAR_FP8_ATTN_SIM=$MODE (rebuilds via bazelw +
# graph compile), renders the 4 prompts (seed 42, 1024x1024) to
# /root/fp8sim/${MODE}_${name}.png, then tears the serve down before the next
# mode. Purely a QUALITY experiment -- 'full' is unfused and slow by design.
set -uo pipefail

MODES=("$@")
[ ${#MODES[@]} -eq 0 ] && MODES=(qkv full)

OUTDIR=/root/fp8sim
mkdir -p "$OUTDIR"
STATUS="$OUTDIR/run.status"
PY=/root/wheeltest-baked/bin/python   # only used as the render HTTP client
: > "$STATUS"

P_portrait='a close-up portrait photo of an elderly woman, deeply wrinkled skin, silver hair, sharp detail, soft studio light'
P_text='a vintage neon storefront sign that reads "OPEN 24 HOURS" glowing at night, reflections on wet pavement'
P_macro='a macro photo of a dragonfly resting on a dew-covered spider web, intricate detail, shallow depth of field'
P_dark='a dimly lit gothic cathedral interior lit only by candlelight, deep shadows, volumetric light rays'

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

serve_one(){
  local mode="$1"
  local serve_log="$OUTDIR/serve_${mode}.log"
  : > "$serve_log"
  pkill -f "pipelines" 2>/dev/null || true
  pkill -f "serve_klein_nvfp4" 2>/dev/null || true
  sleep 4
  log "[$mode] launching serve (MODULAR_FP8_ATTN_SIM=$mode)"
  setsid env MODULAR_NVFP4_W4A4=1 MODULAR_FP8_ATTN_SIM="$mode" \
    bash /root/serve_klein_nvfp4.sh >>"$serve_log" 2>&1 </dev/null &
  local cpid=$!
  sleep 1
  local pgid; pgid=$(ps -o pgid= -p "$cpid" 2>/dev/null | tr -d ' '); [ -z "$pgid" ] && pgid="$cpid"
  echo "$pgid" > "$OUTDIR/${mode}.pgid"
  local ready=0 i
  for i in $(seq 1 1800); do
    if grep -q "Server ready" "$serve_log" 2>/dev/null; then ready=1; log "[$mode] READY after ~${i}s"; break; fi
    if grep -qiE "symbol not found|Worker crashed|Traceback \(most|ABORT:|OUT_OF_MEMORY|unknown op|no operation|CUDA_ERROR|failed to allocate" "$serve_log" 2>/dev/null; then
      log "[$mode] SERVE ERROR at ${i}s:"; tail -30 "$serve_log" | tee -a "$STATUS"; return 11
    fi
    sleep 1
  done
  [ "$ready" = 1 ] || { log "[$mode] NOT READY in 1800s"; tail -30 "$serve_log" | tee -a "$STATUS"; return 10; }
  sleep 3
  return 0
}

teardown(){
  local mode="$1"
  local pgid; pgid=$(cat "$OUTDIR/${mode}.pgid" 2>/dev/null || echo "")
  log "[$mode] teardown pgid=$pgid"
  [ -n "$pgid" ] && kill -TERM -- -"$pgid" 2>/dev/null || true
  sleep 3
  pkill -f "pipelines" 2>/dev/null || true
  sleep 2
}

render_one(){
  local mode="$1" name="$2" prompt="$3"
  local out="$OUTDIR/${mode}_${name}.png"
  rm -f "$out"
  local t0 t1; t0=$(date +%s)
  if "$PY" /root/render_klein.py "$out" "$prompt" >>"$STATUS" 2>&1; then
    t1=$(date +%s); log "[$mode] render $name OK size=$(stat -c %s "$out" 2>/dev/null || echo 0) took=$((t1-t0))s"
  else
    local rc=$?; t1=$(date +%s); log "[$mode] render $name FAILED rc=$rc took=$((t1-t0))s"
  fi
}

for mode in "${MODES[@]}"; do
  log "==== MODE $mode start ===="
  if serve_one "$mode"; then
    render_one "$mode" portrait "$P_portrait"
    render_one "$mode" text     "$P_text"
    render_one "$mode" macro    "$P_macro"
    render_one "$mode" dark     "$P_dark"
  else
    log "[$mode] serve failed, skipping renders"
  fi
  teardown "$mode"
  log "==== MODE $mode done ===="
done
log "ALLDONE"

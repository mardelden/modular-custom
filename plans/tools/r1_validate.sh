#!/usr/bin/env bash
# R1 validation: serve NVFP4 Klein with the FUSED weight_scale_2 epilogue,
# render the 4 standard prompts (seed 42, 1024^2), byte-compare against the
# known-good W4A4 renders (must be IDENTICAL), then time N macro renders
# (scheduler Starting->Completed) for the before/after latency delta.
set -uo pipefail

N=${1:-8}
OUT=/root/r1fused
mkdir -p "$OUT"
STATUS="$OUT/run.status"
SLOG="$OUT/serve.log"
PY=/root/wheeltest-baked/bin/python
BAK=/root/w4a4_naive_kernel_renders
: > "$STATUS"; : > "$SLOG"

P_portrait='a close-up portrait photo of an elderly woman, deeply wrinkled skin, silver hair, sharp detail, soft studio light'
P_text='a vintage neon storefront sign that reads "OPEN 24 HOURS" glowing at night, reflections on wet pavement'
P_macro='a macro photo of a dragonfly resting on a dew-covered spider web, intricate detail, shallow depth of field'
P_dark='a dimly lit gothic cathedral interior lit only by candlelight, deep shadows, volumetric light rays'

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }
log "R1 fused-ws2 validation start"

pkill -f "pipelines" 2>/dev/null || true; sleep 4
setsid env MODULAR_NVFP4_W4A4=1 bash /root/serve_klein_nvfp4.sh >>"$SLOG" 2>&1 </dev/null &
CPID=$!; sleep 1
PGID=$(ps -o pgid= -p "$CPID" 2>/dev/null | tr -d ' '); [ -z "$PGID" ] && PGID="$CPID"

READY=0
for i in $(seq 1 2400); do
  grep -q "Server ready" "$SLOG" 2>/dev/null && { READY=1; log "READY ~${i}s"; break; }
  grep -qiE "symbol not found|Worker crashed|Traceback \(most|ABORT:|unknown op|no operation|CUDA_ERROR|ILLEGAL|Assert Error|failed to legalize|error:" "$SLOG" 2>/dev/null && {
    log "SERVE ERROR at ${i}s:"; tail -30 "$SLOG" | tee -a "$STATUS"; log "ALLDONE rc=11"; exit 11; }
  sleep 1
done
[ "$READY" = 1 ] || { log "NOT READY"; tail -30 "$SLOG" | tee -a "$STATUS"; log "ALLDONE rc=10"; exit 10; }
sleep 3

render(){
  local name="$1" prompt="$2"; local out="$OUT/${name}.png"; rm -f "$out"
  if "$PY" /root/render_klein.py "$out" "$prompt" >>"$STATUS" 2>&1; then
    log "render $name OK size=$(stat -c %s "$out" 2>/dev/null||echo 0)"
  else
    log "render $name FAILED rc=$?"
  fi
}
render portrait "$P_portrait"
render text     "$P_text"
render macro    "$P_macro"
render dark     "$P_dark"

# ---- byte-identity gate vs known-good W4A4 renders ----
IDENT_OK=1
for name in portrait text macro dark; do
  if cmp -s "$OUT/${name}.png" "$BAK/w4a4_${name}.png"; then
    log "IDENTITY $name: BYTE-IDENTICAL"
  else
    log "IDENTITY $name: DIFFERS (fused=$(stat -c %s "$OUT/${name}.png" 2>/dev/null||echo 0) baseline=$(stat -c %s "$BAK/w4a4_${name}.png" 2>/dev/null||echo 0))"
    IDENT_OK=0
  fi
done

# ---- timing: N macro renders, steady-state ----
for k in $(seq 1 "$N"); do
  "$PY" /root/render_klein.py "$OUT/t_${k}.png" "$P_macro" >/dev/null 2>&1 || true
done
"$PY" - "$SLOG" <<"PYEOF" | tee -a "$STATUS"
import re, sys
def ms(t):
    h,m,s = t.split(":"); return (int(h)*3600+int(m)*60+float(s))*1000.0
starts=[]; comps=[]
for ln in open(sys.argv[1]):
    mt = re.search(r"(\d\d:\d\d:\d\d\.\d+).*OneShotScheduler: (Starting|Completed)", ln)
    if not mt: continue
    (starts if mt.group(2)=="Starting" else comps).append(ms(mt.group(1)))
d=[c-s for s,c in zip(starts,comps)]
tail=sorted(d[-8:]) if len(d)>=8 else sorted(d[1:] or d)
med=tail[len(tail)//2]
print("[r1] renders=%d steady8(ms): median=%.0f min=%.0f max=%.0f" % (len(d), med, tail[0], tail[-1]))
PYEOF

log "teardown"; kill -TERM -- -"$PGID" 2>/dev/null || true; sleep 3; pkill -f pipelines 2>/dev/null || true
if [ "$IDENT_OK" = 1 ]; then log "ALLDONE rc=0 IDENTITY=PASS"; else log "ALLDONE rc=2 IDENTITY=FAIL"; fi

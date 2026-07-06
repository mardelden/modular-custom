#!/usr/bin/env bash
# K4: A/B render latency. NVFP4 Klein render (W4A4 GEMM) WITH fp8 attention vs
# WITH bf16 attention. For each config: serve, render N times, teardown. Timing
# from the OneShotScheduler "Starting"/"Completed" server-side timestamps
# (excludes HTTP/network). Reports steady-state (drop the warmup render).
set -uo pipefail

N=${1:-8}
OUT=/root/fp8ab
mkdir -p "$OUT"
STATUS="$OUT/run.status"
PY=/root/wheeltest-baked/bin/python
: > "$STATUS"
PROMPT='a macro photo of a dragonfly resting on a dew-covered spider web, intricate detail, shallow depth of field'

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

run_cfg(){
  local tag="$1"; shift            # remaining args = env assignments
  local slog="$OUT/serve_${tag}.log"; : > "$slog"
  pkill -f "pipelines" 2>/dev/null || true; sleep 4
  log "[$tag] serve env: $*"
  setsid env "$@" bash /root/serve_klein_nvfp4.sh >>"$slog" 2>&1 </dev/null &
  local cpid=$!; sleep 1
  local pgid; pgid=$(ps -o pgid= -p "$cpid" 2>/dev/null|tr -d ' '); [ -z "$pgid" ] && pgid="$cpid"
  local i ready=0
  for i in $(seq 1 2400); do
    grep -q "Server ready" "$slog" 2>/dev/null && { ready=1; log "[$tag] READY ~${i}s"; break; }
    grep -qiE "CUDA_ERROR|ILLEGAL|Worker crashed|Traceback|Assert Error|symbol not found" "$slog" 2>/dev/null && { log "[$tag] SERVE ERR"; tail -20 "$slog"|tee -a "$STATUS"; return 11; }
    sleep 1
  done
  [ "$ready" = 1 ] || { log "[$tag] NOT READY"; return 10; }
  sleep 3
  local k
  for k in $(seq 1 "$N"); do
    "$PY" /root/render_klein.py "$OUT/${tag}_${k}.png" "$PROMPT" >/dev/null 2>&1 || log "[$tag] render $k failed"
  done
  # server-side per-render ms from scheduler Starting/Completed pairs
  "$PY" - "$slog" "$tag" <<"PY" | tee -a "$STATUS"
import re, sys
log, tag = sys.argv[1], sys.argv[2]
def ms(t):
    h,m,s = t.split(":"); return (int(h)*3600+int(m)*60+float(s))*1000.0
starts=[]; comps=[]
for ln in open(log):
    mt = re.search(r"(\d\d:\d\d:\d\d\.\d+).*OneShotScheduler: (Starting|Completed)", ln)
    if not mt: continue
    (starts if mt.group(2)=="Starting" else comps).append(ms(mt.group(1)))
d=[c-s for s,c in zip(starts,comps)]
steady=d[1:] if len(d)>1 else d
steady=sorted(steady)
med = steady[len(steady)//2] if steady else float("nan")
print("[%s] renders=%d steady(ms): median=%.0f min=%.0f max=%.0f  all=%s" % (
    tag, len(d), med, min(steady), max(steady), ",".join("%.0f"%x for x in steady)))
PY
  log "[$tag] teardown"; kill -TERM -- -"$pgid" 2>/dev/null || true; sleep 3; pkill -f pipelines 2>/dev/null || true; sleep 2
}

log "==== A/B start (N=$N) ===="
run_cfg fp8  MODULAR_NVFP4_W4A4=1 MODULAR_NVFP4_FP8_ATTN=1
run_cfg bf16 MODULAR_NVFP4_W4A4=1
log "ALLDONE"

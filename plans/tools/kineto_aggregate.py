#!/usr/bin/env python3
"""Aggregate a libkineto Chrome trace of one Klein render into ranked GPU sinks.

Buckets GPU kernel time by name pattern (W4A4 GEMM, activation-quant, mha,
matmul-other, conv/VAE, elementwise+norm+rope, memcpy, other), reports GPU busy
vs wall (launch-gap share), event counts, and the top-N kernels by total time.

Usage: python kineto_aggregate.py /path/to/trace.json [topN]
"""
import json
import re
import sys
from collections import defaultdict

path = sys.argv[1]
topn = int(sys.argv[2]) if len(sys.argv) > 2 else 30

with open(path) as f:
    data = json.load(f)
events = data.get("traceEvents", data if isinstance(data, list) else [])

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}
kern = defaultdict(lambda: [0.0, 0])   # name -> [total_us, count]
cats = defaultdict(lambda: [0.0, 0])
gpu_events = []
for e in events:
    if e.get("ph") != "X":
        continue
    cat = str(e.get("cat", "")).lower()
    if cat not in GPU_CATS:
        continue
    dur = float(e.get("dur", 0.0))
    name = e.get("name", "?")
    kern[name][0] += dur
    kern[name][1] += 1
    cats[cat][0] += dur
    cats[cat][1] += 1
    gpu_events.append((float(e["ts"]), dur))

if not gpu_events:
    print("NO GPU EVENTS FOUND. cats present:",
          sorted({e.get("cat") for e in events if e.get("ph") == "X"})[:20])
    sys.exit(1)

t0 = min(ts for ts, _ in gpu_events)
t1 = max(ts + d for ts, d in gpu_events)
wall_ms = (t1 - t0) / 1000.0
busy_ms = sum(d for _, d in gpu_events) / 1000.0

# merged-interval busy (true GPU-occupied wall, handles overlap)
iv = sorted((ts, ts + d) for ts, d in gpu_events)
merged = 0.0
cs, ce = iv[0]
for s, e2 in iv[1:]:
    if s <= ce:
        ce = max(ce, e2)
    else:
        merged += ce - cs
        cs, ce = s, e2
merged += ce - cs
merged_ms = merged / 1000.0

BUCKETS = [
    ("mha/attention",      r"mha|flash"),
    ("W4A4 gemm",          r"w4a4|nvfp4|block_scaled"),
    ("act-quant",          r"quant"),
    ("matmul-other(bf16)", r"matmul|gemm"),
    ("conv (VAE)",         r"conv"),
    ("norm/rope/elemwise", r"norm|rope|elementwise|elemwise|silu|gelu|softmax|fused|add|mul|sub|cast|pointwise|broadcast|transpose|concat|split|copy|pad|arange|iota|reduce"),
    ("memcpy/memset",      r"memcpy|memset|__memcpy|CUDA mem"),
]
bucket_tot = defaultdict(lambda: [0.0, 0])
name_bucket = {}
for name, (tot, cnt) in kern.items():
    b = "other"
    low = name.lower()
    for bname, pat in BUCKETS:
        if re.search(pat, low):
            b = bname
            break
    name_bucket[name] = b
    bucket_tot[b][0] += tot
    bucket_tot[b][1] += cnt

print(f"trace: {path}")
print(f"GPU window wall: {wall_ms:8.1f} ms")
print(f"GPU busy (merged intervals): {merged_ms:8.1f} ms  ({100*merged_ms/wall_ms:.1f}% of wall; gap {wall_ms-merged_ms:.1f} ms)")
print(f"GPU busy (summed durations): {busy_ms:8.1f} ms  (>merged means stream overlap)")
print(f"total GPU events: {sum(c for _, c in cats.values())}")
print()
print("=== bucket summary (by summed kernel time) ===")
for b, (tot, cnt) in sorted(bucket_tot.items(), key=lambda kv: -kv[1][0]):
    print(f"  {b:20} {tot/1000.0:9.1f} ms  {100*tot/1000.0/busy_ms:5.1f}%  n={cnt}")
print()
print(f"=== top {topn} kernels ===")
for name, (tot, cnt) in sorted(kern.items(), key=lambda kv: -kv[1][0])[:topn]:
    short = name if len(name) <= 100 else name[:97] + "..."
    print(f"  {tot/1000.0:9.2f} ms  n={cnt:5}  [{name_bucket[name]}]  {short}")

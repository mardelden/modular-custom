#!/usr/bin/env bash
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# ===----------------------------------------------------------------------=== #
#
# Run the Whisper word-timestamp parity gates on real hardware (deploy loop).
#
#   PY=/root/wheeltest-baked/bin/python \
#   MODEL=openai/whisper-large-v3 DEVICE=gpu AUDIO=/root/jfk.wav \
#   PYTHONPATH=/opt/modular-whisper/max/python \
#   ./run_gates.sh
#
# Env:
#   PY        python interpreter (must import `max` + transformers)   [python]
#   MODEL     HF whisper repo                          [openai/whisper-large-v3]
#   DEVICE    cpu | gpu                                                    [gpu]
#   AUDIO     path to a <=30s wav (enables gates c/d; skipped if empty)     []
#   PYTHONPATH must point at the worktree's max/python so the whisper code
#             overlays the installed wheel.
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}
MODEL=${MODEL:-openai/whisper-large-v3}
DEVICE=${DEVICE:-gpu}
AUDIO=${AUDIO:-}
rc=0

run() { echo; echo "== $1 =="; shift; "$@" || rc=1; }

run "gate (a) encoder"   "$PY" parity_encoder.py  --model "$MODEL" --device "$DEVICE" --random
run "gate (b) decoder"   "$PY" parity_decoder.py  --model "$MODEL" --device "$DEVICE"
run "timing unit"        "$PY" check_timing.py    --model "$MODEL"
run "pipeline wiring"    "$PY" check_pipeline.py  --model "$MODEL" --device "$DEVICE"

if [ -n "$AUDIO" ]; then
  run "gate (c) transcript" "$PY" parity_transcript.py --audio "$AUDIO" --model "$MODEL" --device "$DEVICE"
  echo; echo "== gate (d) word timestamps =="
  "$PY" -m max.pipelines.architectures.whisper.cli "$AUDIO" --model "$MODEL" --device "$DEVICE" -o /tmp/max_words.json || rc=1
  "$PY" dump_hf_ref.py --audio "$AUDIO" --model "$MODEL" -o /tmp/hf_ref.json || rc=1
  "$PY" parity_words.py --hyp /tmp/max_words.json --ref /tmp/hf_ref.json --tol-ms 80 || rc=1
  echo "(acceptance band vs faster-whisper: run dump_faster_whisper_ref.py in a"
  echo " throwaway venv, then parity_words.py --hyp /tmp/max_words.json --ref /tmp/fw_ref.json)"
else
  echo; echo "(set AUDIO=<wav> to also run gates c/d)"
fi

echo; echo "==== overall: $([ $rc -eq 0 ] && echo PASS || echo FAIL) ===="
exit $rc

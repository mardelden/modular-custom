#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""End-to-end wiring smoke: WhisperTranscriber on synthetic audio.

Validates that the full pipeline COMPOSES and runs without crashing — weight
loading (safetensors path), the 3 graphs, the greedy loop + suppression, the
align pass, and word timing all integrate and return a well-formed result. The
transcript itself is meaningless (synthetic audio); real transcript/timestamp
accuracy is the on-hardware gate. Token budget is capped to keep it quick.

    python check_pipeline.py --model openai/whisper-tiny --device cpu
"""

from __future__ import annotations

import argparse
import wave

import numpy as np


def write_synth_wav(path: str, seconds: float = 3.0, sr: int = 16000) -> None:
    rng = np.random.RandomState(0)
    # Low-amplitude noise + a couple of tones (keeps the encoder from NaNing).
    t = np.arange(int(seconds * sr)) / sr
    sig = 0.02 * rng.randn(t.size) + 0.05 * np.sin(2 * np.pi * 220 * t)
    pcm = (np.clip(sig, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-tiny")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args()

    from max.pipelines.architectures.whisper.transcribe import (
        WhisperTranscriber,
    )

    wav = "/tmp/whisper_synth.wav"
    write_synth_wav(wav)

    print(f"Loading transcriber ({args.model}, {args.device}) ...")
    tr = WhisperTranscriber(args.model, device=args.device)
    result = tr.transcribe(
        wav, word_timestamps=True, max_new_tokens=args.max_new_tokens
    )

    assert isinstance(result, dict) and "text" in result and "words" in result
    print(f"  text  = {result['text']!r}")
    print(f"  words = {len(result['words'])}")
    for w in result["words"][:8]:
        assert {"word", "start", "end", "probability"} <= set(w)
        assert w["end"] >= w["start"] >= 0.0
        print(f"    {w['start']:6.2f}-{w['end']:6.2f}  {w['word']!r}")
    starts = [w["start"] for w in result["words"]]
    assert starts == sorted(starts), "word starts must be monotonic"
    print("[PASS] end-to-end pipeline composes + returns well-formed words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

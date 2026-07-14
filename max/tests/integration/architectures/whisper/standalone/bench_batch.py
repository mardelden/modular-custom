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
"""Gate Batch: batched decode == per-clip decode, + throughput scaling.

    python bench_batch.py --model openai/whisper-large-v3 --device gpu \\
        --audio-dir /root/clips --batches 1,4,8,16

With no --audio-dir, generates distinct synthetic clips (parity still holds;
throughput scaling only shows on GPU).
"""

from __future__ import annotations

import argparse
import glob
import os
import time
import wave

import numpy as np


def synth(path: str, seed: int, seconds: float = 4.0, sr: int = 16000) -> None:
    rng = np.random.RandomState(seed)
    t = np.arange(int(seconds * sr)) / sr
    tone = 0.05 * np.sin(2 * np.pi * (180 + 40 * seed) * t)
    pcm = (np.clip(0.02 * rng.randn(t.size) + tone, -1, 1) * 32767).astype(
        np.int16
    )
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def words_match(a: list[dict], b: list[dict], tol: float = 1e-3) -> bool:
    return len(a) == len(b) and all(
        x["word"] == y["word"]
        and abs(x["start"] - y["start"]) < tol
        and abs(x["end"] - y["end"]) < tol
        for x, y in zip(a, b)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-tiny")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--audio-dir", default="")
    ap.add_argument("--batches", default="1,4,8")
    args = ap.parse_args()
    batch_sizes = [int(b) for b in args.batches.split(",")]
    n_clips = max(batch_sizes)

    from max.pipelines.architectures.whisper.transcribe import (
        WhisperTranscriber,
    )

    if args.audio_dir:
        clips = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))[
            :n_clips
        ]
        if len(clips) < n_clips:
            raise SystemExit(
                f"need {n_clips} wavs in {args.audio_dir}, found {len(clips)}"
            )
    else:
        clips = [f"/tmp/whisper_batch_{i}.wav" for i in range(n_clips)]
        for i, c in enumerate(clips):
            synth(c, seed=i)
        print(f"(no --audio-dir; generated {n_clips} synthetic clips)")

    tr = WhisperTranscriber(args.model, device=args.device, use_kv_cache=True)

    # Per-clip reference (batch-1).
    ref = [tr.transcribe(c, word_timestamps=True) for c in clips]

    # Parity: each row of a batch matches its standalone transcription.
    all_ok = True
    for bsz in batch_sizes:
        res = tr.transcribe_batch(clips[:bsz], word_timestamps=True)
        ok = all(
            res[b]["text"] == ref[b]["text"]
            and words_match(res[b]["words"], ref[b]["words"])
            for b in range(bsz)
        )
        all_ok = all_ok and ok
        print(f"[{'PASS' if ok else 'FAIL'}] batch={bsz} matches per-clip")

    # Throughput (one warm run already done above).
    print("\nthroughput:")
    for bsz in batch_sizes:
        t0 = time.perf_counter()
        tr.transcribe_batch(clips[:bsz], word_timestamps=True)
        dt = time.perf_counter() - t0
        print(f"  batch={bsz:2d}: {dt * 1e3:7.0f}ms  {bsz / dt:6.2f} clips/s")

    print("GATE BATCH PASS" if all_ok else "GATE BATCH FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

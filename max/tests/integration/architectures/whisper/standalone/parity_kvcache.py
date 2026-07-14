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
"""Gate K: KV-cached decode == no-cache decode (transcript + words), + speedup.

python parity_kvcache.py --model openai/whisper-large-v3 --device gpu --audio clip.wav
"""

from __future__ import annotations

import argparse
import time
import wave

import numpy as np


def write_synth_wav(path: str, seconds: float = 4.0, sr: int = 16000) -> None:
    rng = np.random.RandomState(0)
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
    ap.add_argument("--audio", default="")
    ap.add_argument("--reps", type=int, default=3, help="decode timing repeats")
    args = ap.parse_args()

    from max.pipelines.architectures.whisper.audio import (
        extract_features,
        load_audio,
    )
    from max.pipelines.architectures.whisper.transcribe import (
        WhisperTranscriber,
    )

    audio = args.audio
    if not audio:
        audio = "/tmp/whisper_kv_synth.wav"
        write_synth_wav(audio)
        print(f"(no --audio; using synthetic {audio})")

    print("loading cached transcriber ...")
    tr_c = WhisperTranscriber(args.model, device=args.device, use_kv_cache=True)
    res_c = tr_c.transcribe(audio, word_timestamps=True)
    print("loading no-cache transcriber ...")
    tr_n = WhisperTranscriber(
        args.model, device=args.device, use_kv_cache=False
    )
    res_n = tr_n.transcribe(audio, word_timestamps=True)

    print(f"cached : {res_c['text']!r}")
    print(f"nocache: {res_n['text']!r}")

    text_ok = res_c["text"] == res_n["text"]
    wc, wn = res_c["words"], res_n["words"]
    words_ok = len(wc) == len(wn) and all(
        a["word"] == b["word"]
        and abs(a["start"] - b["start"]) < 1e-3
        and abs(a["end"] - b["end"]) < 1e-3
        for a, b in zip(wc, wn)
    )
    print(f"[{'PASS' if text_ok else 'FAIL'}] transcript identical")
    print(
        f"[{'PASS' if words_ok else 'FAIL'}] words identical "
        f"({len(wc)} vs {len(wn)})"
    )

    # Decode-only timing (shared mel/enc), averaged over reps.
    mel, _ = extract_features(load_audio(audio), tr_c.feature_extractor)
    enc_c = tr_c._encode(mel)
    enc_n = tr_n._encode(mel)

    def timeit(fn, enc):
        best = min(
            (
                lambda: (
                    (t := time.perf_counter(), fn(enc), time.perf_counter())[2]
                    - t
                )
            )()
            for _ in range(args.reps)
        )
        return best

    t_cached = timeit(tr_c._greedy_decode_cached, enc_c)
    t_nocache = timeit(tr_n._greedy_decode, enc_n)
    speedup = t_nocache / t_cached if t_cached else float("nan")
    print(
        f"decode: cached={t_cached * 1e3:.0f}ms  nocache={t_nocache * 1e3:.0f}ms "
        f"speedup={speedup:.2f}x  ({len(wc)} words)"
    )

    ok = text_ok and words_ok
    print("GATE K PASS" if ok else "GATE K FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

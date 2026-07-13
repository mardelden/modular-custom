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
"""Dump a faster-whisper word-timestamp reference to JSON (acceptance ref).

Run in a throwaway venv so it doesn't perturb the MAX env:
    uv venv /tmp/fw && /tmp/fw/bin/pip install faster-whisper
    /tmp/fw/bin/python dump_faster_whisper_ref.py --audio jfk.wav \\
        --model large-v3 -o /tmp/fw_ref.json

Output schema matches the MAX CLI: {"text", "words":[{word,start,end,probability}]}.
"""

from __future__ import annotations

import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--language", default="en")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device="auto", compute_type="default")
    segments, _ = model.transcribe(
        args.audio,
        language=args.language,
        word_timestamps=True,
        beam_size=1,
        temperature=0,
    )

    words = []
    texts = []
    for seg in segments:
        texts.append(seg.text)
        for w in seg.words or []:
            words.append(
                {
                    "word": w.word,
                    "start": round(float(w.start), 3),
                    "end": round(float(w.end), 3),
                    "probability": round(float(w.probability), 4),
                }
            )
    out = {"text": "".join(texts), "words": words}
    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"wrote {args.output}: {len(words)} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
"""Dump HF word timestamps (``return_timestamps="word"``) to JSON — primary ref
for gate (d). Same env as MAX (transformers is already present).

    python dump_hf_ref.py --audio jfk.wav --model openai/whisper-large-v3 -o hf_ref.json
"""

from __future__ import annotations

import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--language", default="en")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    from max.pipelines.architectures.whisper.audio import load_audio
    from transformers import pipeline

    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        chunk_length_s=30,
    )
    audio = load_audio(args.audio)
    out = pipe(
        audio,
        return_timestamps="word",
        generate_kwargs={"language": args.language, "task": "transcribe"},
    )
    words = [
        {
            "word": c["text"],
            "start": round(float(c["timestamp"][0]), 3),
            "end": round(float(c["timestamp"][1] or c["timestamp"][0]), 3),
            "probability": None,
        }
        for c in out.get("chunks", [])
    ]
    result = {"text": out.get("text", ""), "words": words}
    with open(args.output, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"wrote {args.output}: {len(words)} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

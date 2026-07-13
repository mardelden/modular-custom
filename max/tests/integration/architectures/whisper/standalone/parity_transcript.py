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
"""Gate (c): MAX greedy transcript vs HF greedy transcript on a real clip.

    python parity_transcript.py --audio jfk.wav --model openai/whisper-large-v3 --device gpu

Run this on real hardware (deploy loop). Passes if the normalized transcripts
match (and reports token-level agreement).
"""

from __future__ import annotations

import argparse
import re


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="path to a <=30s wav")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--language", default="en")
    args = ap.parse_args()

    import torch
    from common import load_hf
    from max.pipelines.architectures.whisper.audio import load_audio
    from max.pipelines.architectures.whisper.transcribe import (
        WhisperTranscriber,
    )

    # MAX side.
    tr = WhisperTranscriber(
        args.model, device=args.device, language=args.language
    )
    max_text = tr.transcribe(args.audio, word_timestamps=False)["text"]

    # HF reference (greedy).
    config, hf_model, processor = load_hf(args.model)
    audio = load_audio(args.audio)
    feats = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features
    with torch.no_grad():
        ids = hf_model.generate(
            feats,
            num_beams=1,
            do_sample=False,
            language=args.language,
            task="transcribe",
            return_timestamps=False,
        )
    hf_text = processor.batch_decode(ids, skip_special_tokens=True)[0]

    print(f"MAX: {max_text!r}")
    print(f"HF : {hf_text!r}")
    ok = normalize(max_text) == normalize(hf_text)
    # Token-level agreement (word overlap) for diagnostics.
    a, b = normalize(max_text).split(), normalize(hf_text).split()
    overlap = sum(x == y for x, y in zip(a, b)) / max(len(b), 1)
    print(
        f"[{'PASS' if ok else 'FAIL'}] transcript match; word_overlap={overlap:.3f}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

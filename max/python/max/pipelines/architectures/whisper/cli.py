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
"""CLI: transcribe an audio file with word-level timestamps.

    python -m max.pipelines.architectures.whisper.cli AUDIO.wav \\
        --model openai/whisper-large-v3 --device gpu --language en -o out.json
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", help="path to a <=30s audio file")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--language", default="en")
    ap.add_argument(
        "--no-words", action="store_true", help="skip word timestamps"
    )
    ap.add_argument("-o", "--output", help="write JSON here (default: stdout)")
    args = ap.parse_args()

    from .transcribe import WhisperTranscriber

    transcriber = WhisperTranscriber(
        args.model, device=args.device, language=args.language
    )
    result = transcriber.transcribe(
        args.audio, word_timestamps=not args.no_words
    )

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"wrote {args.output} ({len(result['words'])} words)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

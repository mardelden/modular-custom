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
"""Gate (d): compare two word-timestamp JSON files (MAX vs a reference).

Venv-agnostic — just compares JSON, so the reference can come from HF
(``--ref hf_ref.json``) or faster-whisper (``dump_faster_whisper_ref.py``).

    python parity_words.py --hyp max_words.json --ref fw_ref.json --tol-ms 80
"""

from __future__ import annotations

import argparse
import json
import re


def norm(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


def load_words(path: str):
    with open(path) as f:
        data = json.load(f)
    return data["words"] if isinstance(data, dict) else data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True, help="MAX words JSON")
    ap.add_argument("--ref", required=True, help="reference words JSON")
    ap.add_argument("--tol-ms", type=float, default=80.0)
    ap.add_argument("--p95-ms", type=float, default=160.0)
    args = ap.parse_args()

    hyp, ref = load_words(args.hyp), load_words(args.ref)
    # Align on the word text sequence (skip words that don't line up 1:1).
    n = min(len(hyp), len(ref))
    d_start, d_end, matched = [], [], 0
    for i in range(n):
        if norm(hyp[i]["word"]) != norm(ref[i]["word"]):
            continue
        matched += 1
        d_start.append(abs(hyp[i]["start"] - ref[i]["start"]) * 1000.0)
        d_end.append(abs(hyp[i]["end"] - ref[i]["end"]) * 1000.0)

    if not d_start:
        print("[FAIL] no aligned words to compare")
        return 1

    import statistics as st

    def p95(xs):
        return sorted(xs)[max(0, int(round(0.95 * len(xs))) - 1)]

    word_match = matched / max(len(ref), 1)
    med = st.median(d_start + d_end)
    p = p95(d_start + d_end)
    print(f"aligned {matched}/{len(ref)} words (match={word_match:.3f})")
    print(
        f"|Δ| median={med:.1f}ms  p95={p:.1f}ms  (tol {args.tol_ms}/{args.p95_ms})"
    )
    ok = word_match >= 0.9 and med <= args.tol_ms and p <= args.p95_ms
    print(f"[{'PASS' if ok else 'FAIL'}] word timestamps within tolerance")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

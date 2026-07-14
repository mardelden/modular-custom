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
"""Backend-agnostic unit checks for the pure-numpy word-timing logic.

Synthesizes sharp cross-attention peaks at known frames for known tokens and
verifies DTW recovers monotonic word times that track the peaks, and that the
recovered words reconstruct the input phrase.

    python check_timing.py --model openai/whisper-tiny
"""

from __future__ import annotations

import argparse

import numpy as np
from max.pipelines.architectures.whisper.timing import (
    dtw,
    find_word_alignment,
    median_filter,
)


def check_dtw() -> None:
    # Perfect diagonal: identity alignment should be recovered.
    cost = np.array([[0, 9, 9], [9, 0, 9], [9, 9, 0]], dtype=np.float64)
    ti, tj = dtw(cost)
    assert ti[0] == 0 and tj[0] == 0
    assert ti[-1] == 2 and tj[-1] == 2
    assert list(ti) == sorted(ti), "text indices must be non-decreasing"
    print("[PASS] dtw diagonal")


def check_median() -> None:
    x = np.array([[0.0, 100.0, 0.0, 0.0, 0.0]])
    y = median_filter(x, 3)
    assert y.shape == x.shape
    assert y[0, 1] == 0.0, "spike should be removed by median filter"
    print("[PASS] median_filter spike removal")


def check_alignment(model: str) -> None:
    from transformers import WhisperTokenizerFast

    tok = WhisperTokenizerFast.from_pretrained(model)
    phrase = " the quick brown fox"
    text_tokens = tok(phrase, add_special_tokens=False).input_ids
    n = len(text_tokens)
    sot_len = 4
    n_heads, content_pos = 2, 1500
    total = sot_len + n + 1  # sot + text + eot

    # Peaks spread across the full frame range as broad Gaussian bumps (single-
    # frame spikes would be erased by the width-7 median filter, as check_median
    # shows). Token i attends around frame f_i. NOTE: because of the causal
    # decoder shift, text_token[i] is localized at sequence position sot_len-1+i
    # (the position predicting it), so place its peak there.
    xs = np.arange(content_pos)
    frames = [int((i + 1) / (n + 1) * content_pos) for i in range(n)]
    probs = np.full((n_heads, total, content_pos), 1e-4, dtype=np.float64)
    for i, f in enumerate(frames):
        probs[:, sot_len - 1 + i, :] += np.exp(-0.5 * ((xs - f) / 6.0) ** 2)

    words = find_word_alignment(
        probs,
        text_tokens,
        tok,
        num_content_frames=2 * content_pos,
        token_probs=np.linspace(0.9, 1.0, n),
        sot_len=sot_len,
    )
    joined = "".join(w["word"] for w in words)
    print(f"  phrase   = {phrase!r}")
    print(f"  words    = {[w['word'] for w in words]}")
    print(f"  starts   = {[w['start'] for w in words]}")
    assert joined == phrase, f"reconstructed {joined!r} != {phrase!r}"
    starts = [w["start"] for w in words]
    assert starts == sorted(starts), "word starts must be monotonic"
    assert all(w["end"] >= w["start"] for w in words), "end >= start"
    # The attention peak (onset precedes it) should fall within each word's
    # [start, end] span, and inter-word spacing should track the peak spacing.
    for i, w in enumerate(words):
        peak = frames[i] * 0.02
        assert w["start"] - 0.1 <= peak <= w["end"] + 0.1, (i, w, peak)
    # Steady-state spacing tracks the 6s peak spacing (word 0 clamps to t=0).
    spacing = np.diff([w["start"] for w in words])
    assert np.all(np.abs(spacing[1:] - 6.0) < 1.0), f"spacing {spacing} != ~6s"
    print(
        "[PASS] find_word_alignment monotonic + brackets peaks + reconstructs"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-tiny")
    args = ap.parse_args()
    check_dtw()
    check_median()
    check_alignment(args.model)
    print("ALL TIMING CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

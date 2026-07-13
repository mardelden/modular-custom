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
"""Word-level timestamp extraction (host-side numpy).

Given the decoder's cross-attention probabilities for the alignment heads, this
recovers per-word start/end times via dynamic time warping — the same technique
as openai-whisper's ``timing.py`` and HF's ``_extract_token_timestamps``,
adapted to consume already-softmaxed probs and to produce word spans with a
self-consistent boundary mapping (a word ends where the next word begins).

Pure numpy + the HF tokenizer; no torch/scipy, so it unit-tests without a model
or a device. The authoritative check is the on-hardware A/B vs faster-whisper.
"""

from __future__ import annotations

import string

import numpy as np

# Two mel frames per encoder position; 100 mel frames per second -> each encoder
# position spans 0.02s.
TIME_PRECISION = 0.02


def median_filter(x: np.ndarray, width: int) -> np.ndarray:
    """Median filter along the last axis with reflect padding (length-preserving)."""
    if width <= 1:
        return x
    pad = width // 2
    padding = [(0, 0)] * (x.ndim - 1) + [(pad, pad)]
    xp = np.pad(x, padding, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(xp, width, axis=-1)
    return np.median(windows, axis=-1)


def dtw(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dynamic time warping. ``cost`` is ``[N, M]`` (pass the negated score).

    Returns ``(text_indices, time_indices)`` — the aligned path, each an array
    over the warping steps.
    """
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    acc[0, 0] = 0.0
    trace = np.full((n + 1, m + 1), -1, dtype=np.int8)
    trace[0, 1:] = 2  # came from the left
    trace[1:, 0] = 1  # came from above

    # Tie-break exactly like openai-whisper's dtw_cpu: strict `<`, and any tie
    # falls through to "advance the time axis" (t=2). This matters in the large
    # flat regions of the attention matrix, where a diagonal/up preference would
    # collapse most tokens onto a single frame.
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c_diag = acc[i - 1, j - 1]
            c_up = acc[i - 1, j]
            c_left = acc[i, j - 1]
            if c_diag < c_up and c_diag < c_left:
                acc[i, j] = cost[i - 1, j - 1] + c_diag
                trace[i, j] = 0
            elif c_up < c_diag and c_up < c_left:
                acc[i, j] = cost[i - 1, j - 1] + c_up
                trace[i, j] = 1
            else:
                acc[i, j] = cost[i - 1, j - 1] + c_left
                trace[i, j] = 2

    i, j = n, m
    text_idx: list[int] = []
    time_idx: list[int] = []
    while i > 0 or j > 0:
        text_idx.append(i - 1)
        time_idx.append(j - 1)
        t = trace[i, j]
        if t == 0:
            i -= 1
            j -= 1
        elif t == 1:
            i -= 1
        else:
            j -= 1
    return np.array(text_idx[::-1]), np.array(time_idx[::-1])


def split_tokens_on_unicode(
    tokens: list[int], tokenizer
) -> tuple[list[str], list[list[int]]]:
    """Split into decodable subword strings, guarding against split multi-byte chars."""
    decoded_full = tokenizer.decode(tokens)
    replacement = "�"
    words: list[str] = []
    word_tokens: list[list[int]] = []
    current: list[int] = []
    offset = 0
    for token in tokens:
        current.append(token)
        decoded = tokenizer.decode(current)
        idx = decoded.find(replacement)
        if idx < 0 or (
            offset + idx < len(decoded_full)
            and decoded_full[offset + idx] == replacement
        ):
            words.append(decoded)
            word_tokens.append(current)
            current = []
            offset += len(decoded)
    return words, word_tokens


def split_tokens_on_spaces(
    tokens: list[int], tokenizer
) -> tuple[list[str], list[list[int]]]:
    """Group subwords into words on leading spaces / punctuation boundaries."""
    subwords, subword_tokens = split_tokens_on_unicode(tokens, tokenizer)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    words: list[str] = []
    word_tokens: list[list[int]] = []
    for subword, toks in zip(subwords, subword_tokens):
        is_special = toks[0] in special_ids
        with_space = subword.startswith(" ")
        is_punct = subword.strip() in string.punctuation
        if is_special or with_space or is_punct or not words:
            words.append(subword)
            word_tokens.append(list(toks))
        else:
            words[-1] = words[-1] + subword
            word_tokens[-1].extend(toks)
    return words, word_tokens


def merge_punctuations(
    words: list[dict],
    prepended: str = "\"'“¿([{-",
    appended: str = "\"'.。,，!！?？:：”)]}、",
) -> list[dict]:
    """Attach leading/trailing punctuation words onto their neighbor word."""
    # Prepend: a punctuation-only word attaches to the following word.
    i = len(words) - 2
    while i >= 0:
        if words[i]["word"].strip() in prepended and i + 1 < len(words):
            words[i + 1]["word"] = words[i]["word"] + words[i + 1]["word"]
            words[i + 1]["start"] = min(
                words[i]["start"], words[i + 1]["start"]
            )
            words[i]["_merged"] = True
        i -= 1
    # Append: a punctuation-only word attaches to the preceding word.
    i = 1
    while i < len(words):
        if words[i]["word"].strip() in appended:
            words[i - 1]["word"] = words[i - 1]["word"] + words[i]["word"]
            words[i - 1]["end"] = max(words[i - 1]["end"], words[i]["end"])
            words[i]["_merged"] = True
        i += 1
    return [w for w in words if not w.get("_merged")]


def find_word_alignment(
    align_probs: np.ndarray,
    text_tokens: list[int],
    tokenizer,
    num_content_frames: int,
    token_probs: np.ndarray | None = None,
    sot_len: int = 4,
    median_width: int = 7,
) -> list[dict]:
    """Recover word timestamps from alignment-head cross-attention probs.

    Args:
        align_probs: ``[n_heads, T_total, S]`` post-softmax probs, where
            ``T_total`` is the full decoded sequence (SOT prompt + text + EOT).
        text_tokens: the transcript tokens (no prompt, no EOT).
        tokenizer: HF Whisper tokenizer (for token->word splitting).
        num_content_frames: number of real mel frames (<= 3000); crops the
            trailing-silence frames before alignment.
        token_probs: optional per-text-token softmax prob, for word confidence.
        sot_len: number of SOT-prompt tokens to drop from ``align_probs`` rows.
        median_width: median-filter width over the frame axis.

    Returns:
        ``[{"word", "start", "end", "probability"}]`` (seconds).
    """
    content_pos = max(1, num_content_frames // 2)
    w = align_probs[:, :, :content_pos].astype(np.float64)

    # Normalize over the token axis, then median-filter over frames.
    mean = w.mean(axis=-2, keepdims=True)
    std = w.std(axis=-2, keepdims=True)
    w = (w - mean) / (std + 1e-10)
    w = median_filter(w, median_width)

    matrix = w.mean(axis=0)  # [T_total, content_pos]
    # Rows for the transcript tokens, accounting for the causal-decoder shift:
    # the cross-attention at sequence position p localizes the token being
    # PREDICTED (seq[p+1]), not the input token seq[p]. So the row that
    # localizes text_token[i] is one position earlier — the row where the model
    # was predicting it. Slice one back to `[sot_len-1 : -2]` (row i -> text
    # token i). Using `[sot_len:-1]` instead makes every timestamp ~one token
    # (~240ms) late — verified against faster-whisper and HF.
    matrix = matrix[sot_len - 1 : -2]
    n_tok = matrix.shape[0]
    if n_tok == 0:
        return []

    text_indices, time_indices = dtw(-matrix)
    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(
        bool
    )
    token_starts = time_indices[jumps] * TIME_PRECISION  # length n_tok
    audio_end = content_pos * TIME_PRECISION

    # Only time as many tokens as we have alignment rows for.
    tokens = list(text_tokens[:n_tok])
    words, word_tokens = split_tokens_on_spaces(tokens, tokenizer)
    lengths = [len(t) for t in word_tokens]
    boundaries = np.concatenate([[0], np.cumsum(lengths)])  # length W+1

    out: list[dict] = []
    for wi, word in enumerate(words):
        start_tok = int(boundaries[wi])
        end_tok = int(boundaries[wi + 1])
        start = float(token_starts[start_tok])
        end = (
            float(token_starts[end_tok])
            if end_tok < n_tok
            else float(audio_end)
        )
        prob = (
            float(np.mean(token_probs[start_tok:end_tok]))
            if token_probs is not None and end_tok > start_tok
            else None
        )
        out.append(
            {
                "word": word,
                "start": round(start, 3),
                "end": round(max(end, start), 3),
                "probability": (round(prob, 4) if prob is not None else None),
            }
        )
    return merge_punctuations(out)

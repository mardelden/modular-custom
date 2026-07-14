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
"""Gate (b): teacher-forced MAX decoder vs HF decoder.

Feeds the SAME encoder states to both (isolating the decoder), then compares
(1) per-position logits + argmax, and (2) the alignment-head cross-attention
probs. Uses a seeded random mel + an arbitrary token prefix (no datasets).

    python parity_decoder.py --model openai/whisper-tiny --device cpu
"""

from __future__ import annotations

import argparse

import numpy as np
from common import (
    causal_mask,
    decoder_state_dict_from_hf,
    load_hf,
    make_input,
    resolve_device,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-tiny")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    args = ap.parse_args()

    import torch
    from max.driver import Buffer
    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef
    from max.pipelines.architectures.whisper.graph import (
        build_decoder_align_graph,
    )
    from transformers import GenerationConfig

    config, hf_model, _ = load_hf(args.model)
    gen = GenerationConfig.from_pretrained(args.model)
    alignment_heads = [tuple(p) for p in gen.alignment_heads]
    print(f"alignment_heads ({len(alignment_heads)}): {alignment_heads}")

    # Same encoder states for both sides (isolates the decoder).
    rng = np.random.RandomState(0)
    mel = rng.randn(1, config.num_mel_bins, 3000).astype(np.float32)
    with torch.no_grad():
        enc = (
            hf_model.model.encoder(torch.from_numpy(mel))
            .last_hidden_state.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    print(f"encoder states: {enc.shape}")

    # Teacher-forced token prefix: SOT sequence + a few arbitrary in-vocab ids.
    sot = [
        config.decoder_start_token_id,
        gen.lang_to_id["<|en|>"],
        gen.task_to_id["transcribe"],
        gen.no_timestamps_token_id,
    ]
    tokens = np.array([sot + [3123, 456, 789, 2000]], dtype=np.int32)
    T = tokens.shape[1]
    positions = np.arange(T, dtype=np.int32)[None, :]
    mask = causal_mask(T)

    # HF reference.
    with torch.no_grad():
        dec = hf_model.model.decoder(
            input_ids=torch.from_numpy(tokens),
            encoder_hidden_states=torch.from_numpy(enc),
            output_attentions=True,
        )
        hf_logits = (
            hf_model.proj_out(dec.last_hidden_state)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        hf_align = np.stack(
            [
                dec.cross_attentions[l][0, h].detach().cpu().numpy()
                for (l, h) in alignment_heads
            ],
            axis=0,
        ).astype(np.float32)

    # MAX align graph.
    device = resolve_device(args.device)
    state_dict = decoder_state_dict_from_hf(hf_model)
    print(f"decoder state_dict: {len(state_dict)} weights")
    graph = build_decoder_align_graph(
        state_dict,
        config,
        DType.float32,
        DeviceRef.from_device(device),
        alignment_heads,
    )
    session = InferenceSession(devices=[device])
    model = session.load(graph, weights_registry=state_dict)

    def buf(a):
        return Buffer.from_numpy(np.ascontiguousarray(a)).to(device)

    outs = model.execute(buf(tokens), buf(positions), buf(mask), buf(enc))
    max_logits = outs[0].to_numpy().astype(np.float32)
    max_align = outs[1].to_numpy().astype(np.float32)
    print(f"logits: max={max_logits.shape} hf={hf_logits.shape}")
    print(f"align:  max={max_align.shape} hf={hf_align.shape}")

    def cosine(a, b):
        a, b = a.ravel(), b.ravel()
        return float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        )

    # Logits parity. argmax agreement is the correctness-critical metric (it
    # decides the greedy transcript); the tight max_abs only holds on CPU, so on
    # GPU (TF32) fall back to a high logit cosine.
    lg_maxabs = float(np.abs(max_logits - hf_logits).max())
    argmax_match = float((max_logits.argmax(-1) == hf_logits.argmax(-1)).mean())
    lg_cos = cosine(max_logits, hf_logits)
    al_maxabs = float(np.abs(max_align - hf_align).max())
    al_cos = cosine(max_align, hf_align)

    logits_ok = argmax_match >= 0.999 and (lg_maxabs < 1e-3 or lg_cos >= 0.9999)
    align_ok = al_maxabs < 1e-3 or al_cos >= 0.999
    print(
        f"[{'PASS' if logits_ok else 'FAIL'}] logits: max_abs={lg_maxabs:.3e} "
        f"argmax_match={argmax_match:.3f} cos={lg_cos:.6f}"
    )
    print(
        f"[{'PASS' if align_ok else 'FAIL'}] alignment probs: "
        f"max_abs={al_maxabs:.3e} cos={al_cos:.6f}"
    )
    return 0 if (logits_ok and align_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())

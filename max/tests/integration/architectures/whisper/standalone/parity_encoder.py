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
"""Gate (a): MAX Whisper encoder vs HF ``model.model.encoder`` last_hidden_state.

python parity_encoder.py --model openai/whisper-tiny --device cpu
"""

from __future__ import annotations

import argparse

import numpy as np
from common import (
    encoder_state_dict_from_hf,
    load_hf,
    make_input,
    report,
    resolve_device,
    sample_mel,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-tiny")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument(
        "--cos-threshold",
        type=float,
        default=0.9999,
        help="Cosine-similarity pass threshold. GPU f32 matmul uses TF32 "
        "(~1e-2 magnitude drift on deep nets), so the tight rtol/atol only "
        "holds on CPU; cosine ~1.0 is the correctness signal on GPU.",
    )
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument(
        "--random",
        action="store_true",
        help="Use seeded random mel instead of real audio (no datasets dep; "
        "still a valid encoder numeric-parity check).",
    )
    args = ap.parse_args()

    import torch
    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef
    from max.pipelines.architectures.whisper.graph import build_encoder_graph

    print(f"Loading HF model {args.model} ...")
    config, hf_model, processor = load_hf(args.model)
    if args.random:
        rng = np.random.RandomState(0)
        mel = rng.randn(args.samples, config.num_mel_bins, 3000).astype(
            np.float32
        )
        print(f"random mel features: shape={mel.shape} dtype={mel.dtype}")
    else:
        mel = sample_mel(processor, n=args.samples)
        print(f"mel features: shape={mel.shape} dtype={mel.dtype}")

    # HF reference: encoder last hidden state.
    with torch.no_grad():
        ref = (
            hf_model.model.encoder(torch.from_numpy(mel))
            .last_hidden_state.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    print(f"HF encoder output: shape={ref.shape}")

    # MAX encoder graph.
    device = resolve_device(args.device)
    state_dict = encoder_state_dict_from_hf(hf_model)
    print(f"state_dict: {len(state_dict)} weights (encoder tower)")

    graph = build_encoder_graph(
        state_dict, config, DType.float32, DeviceRef.from_device(device)
    )
    session = InferenceSession(devices=[device])
    model = session.load(graph, weights_registry=state_dict)

    out = model.execute(make_input(mel, device))[0]
    got = out.to_numpy().astype(np.float32)
    print(f"MAX encoder output: shape={got.shape}")

    if got.shape != ref.shape:
        print(f"SHAPE MISMATCH: max={got.shape} hf={ref.shape}")
        return 1

    diff = np.abs(got - ref)
    max_abs = float(diff.max())
    denom = float(np.abs(ref).max()) + 1e-12
    rel = max_abs / denom
    cos = float(
        np.dot(got.ravel(), ref.ravel())
        / (np.linalg.norm(got.ravel()) * np.linalg.norm(ref.ravel()) + 1e-12)
    )
    # Pass on either the tight tolerance (CPU full-precision) OR high cosine
    # similarity (GPU TF32 preserves direction but drifts in magnitude).
    strict = bool(np.allclose(got, ref, rtol=args.rtol, atol=args.atol))
    ok = strict or cos >= args.cos_threshold
    report("encoder", max_abs, rel, cos, ok)
    print(
        f"    strict(rtol/atol)={strict}  cos>={args.cos_threshold}:{cos >= args.cos_threshold}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

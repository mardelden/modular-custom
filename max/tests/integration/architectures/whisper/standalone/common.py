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
"""Shared helpers for the standalone Whisper parity scripts.

These are plain scripts (NOT bazel tests — the bazel glob is ``**/test_*.py``,
so ``parity_*``/``dump_*``/``common`` are ignored). They may import torch and
transformers directly. Run them against a venv that has ``modular`` (nightly)
plus ``torch``/``transformers``/``datasets``/``numpy`` installed, with the
whisper package overlaid onto the installed ``max`` (see the plan doc).
"""

from __future__ import annotations

import numpy as np


def resolve_device(name: str):
    """Return a MAX device for ``"cpu"`` or ``"gpu"``."""
    from max.driver import CPU, Accelerator, accelerator_count

    if name == "gpu":
        if accelerator_count() == 0:
            raise RuntimeError(
                "--device gpu requested but no accelerator found"
            )
        return Accelerator()
    return CPU()


def load_hf(model_id: str):
    """Load the HF config, seq2seq model (eval, float32) and processor."""
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
    )

    config = AutoConfig.from_pretrained(model_id)
    # eager attention so `output_attentions=True` materializes the cross-attention
    # weights (SDPA, the transformers-5 default, returns None for attentions).
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return config, model, processor


def sample_mel(processor, n: int = 1) -> np.ndarray:
    """Return ``[n, num_mel_bins, 3000]`` float32 mel features from LibriSpeech-dummy."""
    from datasets import load_dataset

    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    arrays = [ds[i]["audio"]["array"] for i in range(n)]
    sr = ds[0]["audio"]["sampling_rate"]
    inputs = processor(
        arrays,
        return_attention_mask=True,
        sampling_rate=sr,
        return_tensors="np",
    )
    return np.asarray(inputs["input_features"], dtype=np.float32)


def encoder_state_dict_from_hf(hf_model) -> dict[str, np.ndarray]:
    """Rename an HF Whisper ``state_dict`` into the MAX encoder FQNs (float32 numpy).

    Reuses the production rename rule so this exercises the real adapter logic.
    """
    from max.pipelines.architectures.whisper.weight_adapters import (
        _rename_encoder_key,
    )

    out: dict[str, np.ndarray] = {}
    for key, tensor in hf_model.state_dict().items():
        max_name = _rename_encoder_key(key)
        if max_name is None:
            continue
        out[max_name] = tensor.detach().cpu().float().numpy()
    return out


def decoder_state_dict_from_hf(hf_model) -> dict[str, np.ndarray]:
    """Rename an HF Whisper ``state_dict`` into the MAX decoder FQNs (float32 numpy)."""
    from max.pipelines.architectures.whisper.weight_adapters import (
        _rename_decoder_key,
    )

    out: dict[str, np.ndarray] = {}
    for key, tensor in hf_model.state_dict().items():
        max_name = _rename_decoder_key(key)
        if max_name is None:
            continue
        out[max_name] = tensor.detach().cpu().float().numpy()
    return out


def causal_mask(seq_len: int) -> np.ndarray:
    """Additive causal mask ``[1, 1, T, T]`` (0 on/below diagonal, large-neg above)."""
    from max.pipelines.architectures.whisper.decoder import NEG_INF

    m = np.triu(np.full((seq_len, seq_len), NEG_INF, dtype=np.float32), k=1)
    return m[None, None, :, :]


def make_input(mel: np.ndarray, device):
    """Wrap a contiguous numpy mel array as a MAX buffer placed on ``device``."""
    from max.driver import Buffer

    return Buffer.from_numpy(np.ascontiguousarray(mel)).to(device)


def report(name: str, max_abs: float, rel: float, cos: float, ok: bool) -> None:
    status = "PASS" if ok else "FAIL"
    print(
        f"[{status}] {name}: max_abs={max_abs:.3e} rel={rel:.3e} cos={cos:.6f}"
    )

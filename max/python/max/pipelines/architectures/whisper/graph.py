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

from __future__ import annotations

from collections.abc import Mapping

from max.driver import DLPackArray
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import WeightData
from transformers import AutoConfig

from .decoder import WhisperDecoder
from .encoder import WhisperEncoder


def build_encoder_graph(
    state_dict: Mapping[str, DLPackArray | WeightData],
    huggingface_config: AutoConfig,
    dtype: DType,
    device: DeviceRef,
) -> Graph:
    """Builds the Whisper audio-encoder graph.

    Input:  mel features ``[batch, num_mel_bins, 3000]`` (float32).
    Output: encoder hidden states ``[batch, 1500, d_model]`` (float32).

    The input lives on ``device`` (not hard-coded CPU) so the graph runs on
    whichever device the session was created with. ``num_mel_bins`` and the
    frame count are static (from the config): the frame count must be
    ``2 * max_source_positions`` so the stride-2 conv output matches the fixed
    ``[max_source_positions, d_model]`` positional table (a symbolic frame dim
    makes that add non-inferable). The feature extractor always pads to 3000.
    """
    num_frames = 2 * huggingface_config.max_source_positions
    input_features_type = TensorType(
        DType.float32,
        shape=["batch_size", huggingface_config.num_mel_bins, num_frames],
        device=device,
    )

    with Graph(
        "whisper_audio_encoder", input_types=[input_features_type]
    ) as graph:
        model = WhisperEncoder(huggingface_config, dtype, device)
        model.load_state_dict(state_dict)
        input_features = graph.inputs[0]
        outputs = model(input_features=input_features.tensor)
        graph.output(*outputs)
    return graph


# Backwards-compatible alias. The parked ``model.py`` (PipelineModel) imports
# ``build_graph``; keep it pointing at the encoder builder.
build_graph = build_encoder_graph


def _decoder_input_types(
    huggingface_config: AutoConfig, device: DeviceRef
) -> list[TensorType]:
    """Shared decoder graph inputs: tokens, positions, causal mask, encoder states.

    ``seq_len`` is symbolic (one compiled graph serves every prefix length); the
    encoder-state length is static (``max_source_positions`` = 1500).
    """
    seq = "seq_len"
    return [
        TensorType(DType.int32, shape=["batch", seq], device=device),
        TensorType(DType.int32, shape=["batch", seq], device=device),
        TensorType(DType.float32, shape=["batch", 1, seq, seq], device=device),
        TensorType(
            DType.float32,
            shape=[
                "batch",
                huggingface_config.max_source_positions,
                huggingface_config.d_model,
            ],
            device=device,
        ),
    ]


def build_decoder_generate_graph(
    state_dict: Mapping[str, DLPackArray | WeightData],
    huggingface_config: AutoConfig,
    dtype: DType,
    device: DeviceRef,
) -> Graph:
    """Decoder graph for the greedy loop: full logits ``[batch, seq_len, vocab]``.

    (The host takes the last position for the next-token argmax. v1 recomputes
    the full prefix each step — no KV cache.)
    """
    with Graph(
        "whisper_decoder_generate",
        input_types=_decoder_input_types(huggingface_config, device),
    ) as graph:
        model = WhisperDecoder(
            huggingface_config, dtype, device, return_alignment=False
        )
        model.load_state_dict(state_dict)
        tokens, positions, mask, encoder_states = (
            graph.inputs[0].tensor,
            graph.inputs[1].tensor,
            graph.inputs[2].tensor,
            graph.inputs[3].tensor,
        )
        outputs = model(tokens, positions, mask, encoder_states)
        graph.output(*outputs)
    return graph


def build_decoder_align_graph(
    state_dict: Mapping[str, DLPackArray | WeightData],
    huggingface_config: AutoConfig,
    dtype: DType,
    device: DeviceRef,
    alignment_heads: list[tuple[int, int]],
) -> Graph:
    """Teacher-forced decoder graph: full logits + alignment-head cross probs.

    Output 0: logits ``[batch, seq_len, vocab]``.
    Output 1: alignment probs ``[n_align_heads, seq_len, 1500]`` (float32).
    """
    with Graph(
        "whisper_decoder_align",
        input_types=_decoder_input_types(huggingface_config, device),
    ) as graph:
        model = WhisperDecoder(
            huggingface_config,
            dtype,
            device,
            return_alignment=True,
            alignment_heads=alignment_heads,
        )
        model.load_state_dict(state_dict)
        tokens, positions, mask, encoder_states = (
            graph.inputs[0].tensor,
            graph.inputs[1].tensor,
            graph.inputs[2].tensor,
            graph.inputs[3].tensor,
        )
        outputs = model(tokens, positions, mask, encoder_states)
        graph.output(*outputs)
    return graph

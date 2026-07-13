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
"""Weight adapters mapping HF Whisper safetensors to MAX module FQNs.

HF checkpoints prefix everything with ``model.`` and split the model into
``model.encoder.*`` and ``model.decoder.*`` towers. These converters strip the
prefix, keep only the requested tower, and rename the sublayers to the FQNs
produced by the MAX modules (``encoder.py`` / ``decoder.py``).

Both converters run explicitly (segment renames on an already-stripped key)
rather than a blind ordered-substring loop, because some keys contain
misleading substrings — e.g. ``model.decoder.layers.0.encoder_attn.*`` contains
``encoder``, so a naive "drop anything containing 'encoder'" rule would wrongly
drop the decoder's cross-attention.
"""

from __future__ import annotations

from collections.abc import Mapping

from max.dtype import DType
from max.graph.weights import WeightData, Weights


def _rename_encoder_key(name: str) -> str | None:
    """Map an HF Whisper key to its MAX encoder FQN, or ``None`` to drop it.

    Only the ``model.encoder.*`` tower is kept; the decoder tower and the tied
    LM head are dropped (the encoder graph does not use them).
    """
    prefix = "model.encoder."
    if not name.startswith(prefix):
        return None
    key = name[len(prefix) :]

    # Final encoder LayerNorm: `layer_norm.` -> `norm.` (matches WhisperEncoder.norm).
    if key.startswith("layer_norm."):
        return "norm." + key[len("layer_norm.") :]

    # conv1/conv2/embed_positions pass through unchanged; the rest are renamed.
    return (
        key.replace(".self_attn_layer_norm.", ".attention_norm.")
        .replace(".self_attn.", ".attention.")
        .replace(".final_layer_norm.", ".mlp_norm.")
        .replace(".fc1.", ".mlp.fc1.")
        .replace(".fc2.", ".mlp.fc2.")
        .replace(".q_proj.", ".wq.")
        .replace(".k_proj.", ".wk.")
        .replace(".v_proj.", ".wv.")
        .replace(".out_proj.", ".wo.")
    )


def convert_safetensor_state_dict_encoder(
    state_dict: Mapping[str, Weights],
    target_dtype: DType = DType.float32,
) -> dict[str, WeightData]:
    """Convert HF Whisper weights into the encoder module's state dict.

    Args:
        state_dict: HF safetensor weights (values are ``Weights`` handles).
        target_dtype: dtype to cast the weights to. Whisper large-v3 ships in
            fp16, so the default float32 build needs an explicit cast.
    """
    new_state_dict: dict[str, WeightData] = {}
    for weight_name, value in state_dict.items():
        max_name = _rename_encoder_key(weight_name)
        if max_name is None:
            continue
        new_state_dict[max_name] = value.data().astype(target_dtype)
    return new_state_dict


def _rename_decoder_key(name: str) -> str | None:
    """Map an HF Whisper key to its MAX decoder FQN, or ``None`` to drop it.

    Only the ``model.decoder.*`` tower is kept. The tied ``proj_out.weight`` and
    the encoder tower are dropped (the decoder reconstructs its LM head from
    ``embed_tokens.weight``). Renames run in an order that respects the
    ``*_layer_norm`` prefixes so ``encoder_attn_layer_norm`` is not mis-split by
    the ``encoder_attn`` rule.
    """
    prefix = "model.decoder."
    if not name.startswith(prefix):
        return None
    key = name[len(prefix) :]

    # Embeddings pass through unchanged.
    if key in ("embed_tokens.weight", "embed_positions.weight"):
        return key

    # Final decoder LayerNorm: `layer_norm.` -> `norm.` (exact prefix, before the
    # per-sublayer *_layer_norm renames below would ever see it).
    if key.startswith("layer_norm."):
        return "norm." + key[len("layer_norm.") :]

    return (
        key.replace(".self_attn_layer_norm.", ".attention_norm.")
        .replace(".encoder_attn_layer_norm.", ".cross_attention_norm.")
        .replace(".self_attn.", ".attention.")
        .replace(".encoder_attn.", ".cross_attention.")
        .replace(".final_layer_norm.", ".mlp_norm.")
        .replace(".fc1.", ".mlp.fc1.")
        .replace(".fc2.", ".mlp.fc2.")
        .replace(".q_proj.", ".wq.")
        .replace(".k_proj.", ".wk.")
        .replace(".v_proj.", ".wv.")
        .replace(".out_proj.", ".wo.")
    )


def convert_safetensor_state_dict_decoder(
    state_dict: Mapping[str, Weights],
    target_dtype: DType = DType.float32,
) -> dict[str, WeightData]:
    """Convert HF Whisper weights into the decoder module's state dict."""
    new_state_dict: dict[str, WeightData] = {}
    for weight_name, value in state_dict.items():
        max_name = _rename_decoder_key(weight_name)
        if max_name is None:
            continue
        new_state_dict[max_name] = value.data().astype(target_dtype)
    return new_state_dict

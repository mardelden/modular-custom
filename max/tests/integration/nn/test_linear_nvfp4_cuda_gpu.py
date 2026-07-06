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
"""Graph-level numeric test for the NVIDIA NVFP4 (W4A16) ``Linear`` path.

The CUDA sibling of ``test_linear_nvfp4_apple_gpu.py``. Builds a single NVFP4
:class:`~max.nn.Linear` (a FLUX.2 transformer Linear shape) and runs it through
the graph on an NVIDIA GPU **without** the SM100 native block-scaled FP4 path
(e.g. sm_120), then compares the output to a reference graph branch that
performs a plain bf16 dense matmul of the *materialized* dequantized weight.
Both outputs go through the same MAX bf16 MMA, so they must agree to bf16-MMA
tolerance.

This exercises the full NVIDIA weight-only lowering chain:
``Linear.__call__ -> linear() -> quantized_matmul() -> _matmul_float4() ->
(CUDA branch) _cuda_weight_only_block_scaled_matmul() ->
mo.matmul.weight.only.block.scaled.cuda -> nvfp4_w4a16_matmul_cuda`` plus the
graph-level ``weight_scale_2`` fold.

Like the Apple path -- and unlike the SM100 path -- the activation stays in
bf16 (it is *not* dynamically quantized to FP4) and the weight block scales are
plain rank-2 ``[N, K // 16]`` (no rank-5 TCGEN05 interleave). The reference
therefore materializes the dequant exactly as the kernel does:
``E2M1[nibble] * |block_scale| * weight_scale_2``.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    accelerator_api,
    accelerator_architecture_name,
    accelerator_count,
)
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Shape, TensorType, TensorValue, ops
from max.graph.weights import WeightData
from max.nn import Linear
from max.nn.kernels import (
    _cuda_w4a4_matmul,
    _cuda_weight_only_block_scaled_matmul,
    _cuda_weight_only_block_scaled_matmul_fused,
)
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

# E2M1 4-bit code -> float value (sign + magnitude baked in), matching
# `linalg/fp4_utils.mojo` E2M1_TO_FLOAT32. Index is the 4-bit nibble.
_E2M1_TO_FLOAT = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)

_SF_VECTOR_SIZE = 16


def _skip_if_not_cuda_w4a16() -> None:
    if accelerator_count() == 0:
        pytest.skip("No GPU available for the CUDA NVFP4 Linear test")
    if accelerator_api() != "cuda":
        pytest.skip("CUDA W4A16 NVFP4 Linear path requires an NVIDIA GPU")
    if accelerator_architecture_name().startswith("sm_10"):
        pytest.skip(
            "SM100 uses the native block-scaled FP4 path, not the W4A16 "
            "materialize->dense path under test"
        )


def _make_nvfp4_config() -> QuantConfig:
    """NVFP4 block-scaled config (block 16 on K), matching FLUX.2-NVFP4."""
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.STATIC,
            dtype=DType.float32,
            block_size=(1, 16),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e4m3fn,
            block_size=(1, 16),
        ),
        mlp_quantized_layers={0},
        attn_quantized_layers=set(),
        embedding_output_dtype=DType.bfloat16,
        format=QuantFormat.NVFP4,
        scales_pre_interleaved=False,
    )


def _pack_fp4_weight(nibbles: np.ndarray) -> np.ndarray:
    """Pack ``[N, K]`` 4-bit codes into ``uint8 [N, K // 2]`` (low nibble first).

    Element ``2*j`` -> ``byte & 0xF``, element ``2*j+1`` -> ``byte >> 4``
    (the kernel's lo-nibble-first convention).
    """
    lo = nibbles[:, 0::2].astype(np.uint8)
    hi = nibbles[:, 1::2].astype(np.uint8)
    return (lo | (hi << np.uint8(4))).astype(np.uint8)


def _materialize_dequant_weight(
    nibbles: np.ndarray,
    scales_fp32: np.ndarray,
    weight_scale_2: float,
) -> np.ndarray:
    """Dense dequantized weight ``[N, K]`` (fp32): ``E2M1 * |scale| * ws2``."""
    _, k = nibbles.shape
    vals = _E2M1_TO_FLOAT[nibbles]  # [N, K]
    block_scale = np.abs(scales_fp32)[:, : (k // _SF_VECTOR_SIZE)]
    block_scale_full = np.repeat(block_scale, _SF_VECTOR_SIZE, axis=1)[:, :k]
    return (vals * block_scale_full * np.float32(weight_scale_2)).astype(
        np.float32
    )


def _fp32_to_fp8_bytes(
    values_fp32: np.ndarray, device: Accelerator, device_ref: DeviceRef
) -> np.ndarray:
    """Round positive fp32 values to float8_e4m3fn and return the raw bytes.

    numpy has no fp8 dtype, so round-trip the values through a tiny cast graph
    on the device. The test scales are exactly fp8-representable, so this is a
    no-op rounding; it exists to produce the canonical fp8 byte encoding the
    graph const expects. (The cast must run on the accelerator -- fp8 is not a
    supported CPU dtype.)
    """
    flat = values_fp32.reshape(-1).astype(np.float32)
    sess = InferenceSession(devices=[device])
    with Graph(
        "fp32_to_fp8",
        input_types=[
            TensorType(DType.float32, (flat.shape[0],), device=device_ref)
        ],
    ) as g:
        (v,) = g.inputs
        assert isinstance(v, TensorValue)
        g.output(ops.cast(v, DType.float8_e4m3fn))
    out = sess.load(g).execute(Buffer.from_numpy(flat).to(device))[0]
    assert isinstance(out, Buffer)
    return (
        np.from_dlpack(out.to(CPU()).view(DType.uint8))
        .copy()
        .reshape(values_fp32.shape)
    )


def test_linear_nvfp4_cuda() -> None:
    """Numeric check: CUDA NVFP4 Linear == bf16 dense matmul of dequant weight."""
    _skip_if_not_cuda_w4a16()

    rng = np.random.default_rng(0)
    # A FLUX.2 transformer block dim: N=out, K=in. K must be a multiple of 16.
    M, N, K = 8, 256, 512

    device = Accelerator(0)
    device_ref = DeviceRef(device.label, device.id)
    quant_config = _make_nvfp4_config()

    # Random 4-bit codes (full 0..15 range) + fp8-exact positive block scales.
    nibbles = rng.integers(0, 16, size=(N, K), dtype=np.uint8)
    packed = _pack_fp4_weight(nibbles)  # [N, K//2] uint8
    scale_k = K // _SF_VECTOR_SIZE
    # Scales in {0.5, 1.0, 1.5, 2.0} -> exactly fp8-e4m3 representable.
    scales_fp32 = rng.integers(1, 5, size=(N, scale_k)).astype(
        np.float32
    ) * np.float32(0.5)
    weight_scale_2 = np.float32(0.0125)
    input_scale = np.float32(1.0)  # cancels on the W4A16 path; value irrelevant.

    scales_fp8_bytes = _fp32_to_fp8_bytes(scales_fp32, device, device_ref)
    scales_fp8_buf = Buffer.from_numpy(scales_fp8_bytes).view(
        DType.float8_e4m3fn, (N, scale_k)
    )
    weight_scale_wd = WeightData(
        scales_fp8_buf, "weight_scale", DType.float8_e4m3fn, Shape((N, scale_k))
    )

    layer = Linear(
        in_dim=K,
        out_dim=N,
        dtype=DType.uint8,
        device=device_ref,
        has_bias=False,
        quant_config=quant_config,
    )
    layer.load_state_dict(
        {
            "weight": packed,  # uint8 [N, K//2]
            "weight_scale": weight_scale_wd,  # fp8 [N, K//16]
            "weight_scale_2": np.array(weight_scale_2, dtype=np.float32),
            "input_scale": np.array(input_scale, dtype=np.float32),
        },
        weight_alignment=1,
    )

    # Dense dequantized weight (fp32) for the reference matmul.
    w_dense = _materialize_dequant_weight(
        nibbles, scales_fp32, float(weight_scale_2)
    )  # [N, K]

    x_fp32 = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)

    session = InferenceSession(devices=[device])
    with Graph(
        "Linear_NVFP4_CUDA_Test",
        input_types=[TensorType(DType.float32, (M, K), device=device_ref)],
    ) as graph:
        (x_in,) = graph.inputs
        assert isinstance(x_in, TensorValue)
        # Cast activation to bf16 in-graph (matches a real bf16 activation).
        x_bf16 = ops.cast(x_in, DType.bfloat16)

        # Path under test: the NVFP4 Linear (CUDA W4A16 lowering).
        out_test = layer(x_bf16)

        # Reference: plain bf16 dense matmul of the materialized dequant weight.
        w_const = ops.constant(w_dense, DType.float32, device=device_ref)
        w_bf16 = ops.cast(w_const, DType.bfloat16)
        out_ref = ops.matmul(x_bf16, ops.transpose(w_bf16, 0, 1))

        graph.output(
            ops.cast(out_test, DType.float32),
            ops.cast(out_ref, DType.float32),
        )

    compiled = session.load(graph, weights_registry=layer.state_dict())

    x_dev = Buffer.from_numpy(x_fp32).to(device)
    got_buf, ref_buf = compiled.execute(x_dev)

    assert isinstance(got_buf, Buffer)
    assert isinstance(ref_buf, Buffer)
    got = np.from_dlpack(got_buf.to(CPU())).astype(np.float32)
    ref = np.from_dlpack(ref_buf.to(CPU())).astype(np.float32)

    assert got.shape == (M, N)
    assert np.isfinite(got).all(), "CUDA NVFP4 Linear output has NaN/Inf"

    # Both go through the MAX bf16 MMA; the materialize->dense kernel dequants
    # the weight to the same bf16 values the reference uses, so the two outputs
    # should agree to bf16-MMA tolerance.
    atol = 1e-2 + 1.6e-2 * np.abs(ref)
    max_err = float(np.max(np.abs(got - ref)))
    assert np.all(np.abs(got - ref) <= atol), (
        f"CUDA NVFP4 Linear mismatch vs bf16 dequant reference: "
        f"max_err={max_err}"
    )


def test_linear_nvfp4_cuda_fused_matches_materialize() -> None:
    """Fused NVFP4 op == materialize NVFP4 op on identical inputs.

    Builds one graph with two branches over the *same* activation, packed FP4
    weight, and FP8 block scales: the Phase C materialize->dense op
    (``mo.matmul.weight.only.block.scaled.cuda``, the correctness ORACLE) and the
    fused decode-in-SMEM op (``...cuda.fused``). Both dequantize the weight to the
    identical bf16 values and feed the identical bf16 tensor-core MMA, so they
    must agree to bf16-MMA tolerance (only the f32 accumulation order differs).
    """
    _skip_if_not_cuda_w4a16()

    rng = np.random.default_rng(1)
    # FLUX.2 transformer block dim: N=out, K=in. K must be a multiple of 16.
    M, N, K = 8, 256, 512

    device = Accelerator(0)
    device_ref = DeviceRef(device.label, device.id)

    # Random 4-bit codes (full 0..15 range) + fp8-exact positive block scales.
    nibbles = rng.integers(0, 16, size=(N, K), dtype=np.uint8)
    packed = _pack_fp4_weight(nibbles)  # [N, K//2] uint8
    scale_k = K // _SF_VECTOR_SIZE
    # Scales in {0.5, 1.0, 1.5, 2.0} -> exactly fp8-e4m3 representable.
    scales_fp32 = rng.integers(1, 5, size=(N, scale_k)).astype(
        np.float32
    ) * np.float32(0.5)
    scales_fp8_bytes = _fp32_to_fp8_bytes(scales_fp32, device, device_ref)

    x_fp32 = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)

    session = InferenceSession(devices=[device])
    with Graph(
        "NVFP4_CUDA_Fused_vs_Materialize",
        input_types=[
            TensorType(DType.float32, (M, K), device=device_ref),
            TensorType(DType.uint8, (N, K // 2), device=device_ref),
            TensorType(DType.float8_e4m3fn, (N, scale_k), device=device_ref),
        ],
    ) as graph:
        x_in, packed_in, scales_in = graph.inputs
        assert isinstance(x_in, TensorValue)
        assert isinstance(packed_in, TensorValue)
        assert isinstance(scales_in, TensorValue)
        # Cast activation to bf16 in-graph (matches a real bf16 activation).
        x_bf16 = ops.cast(x_in, DType.bfloat16)

        # Oracle: the Phase C materialize->dense NVFP4 op.
        out_mat = _cuda_weight_only_block_scaled_matmul(
            x_bf16, packed_in, scales_in, out_type=DType.bfloat16
        )
        # Path under test: the fused decode-in-SMEM NVFP4 op.
        out_fused = _cuda_weight_only_block_scaled_matmul_fused(
            x_bf16, packed_in, scales_in, out_type=DType.bfloat16
        )

        graph.output(
            ops.cast(out_mat, DType.float32),
            ops.cast(out_fused, DType.float32),
        )

    compiled = session.load(graph)

    x_dev = Buffer.from_numpy(x_fp32).to(device)
    packed_dev = Buffer.from_numpy(packed).to(device)
    scales_dev = (
        Buffer.from_numpy(scales_fp8_bytes)
        .view(DType.float8_e4m3fn, (N, scale_k))
        .to(device)
    )
    mat_buf, fused_buf = compiled.execute(x_dev, packed_dev, scales_dev)

    assert isinstance(mat_buf, Buffer)
    assert isinstance(fused_buf, Buffer)
    mat = np.from_dlpack(mat_buf.to(CPU())).astype(np.float32)
    fused = np.from_dlpack(fused_buf.to(CPU())).astype(np.float32)

    assert fused.shape == (M, N)
    assert np.isfinite(fused).all(), "Fused NVFP4 output has NaN/Inf"

    # Both feed identical bf16 weight values into the same bf16 tensor-core MMA;
    # only the f32 accumulation order differs (different tiling), so they agree
    # to bf16-MMA tolerance. The materialize kernel is the correctness oracle.
    atol = 1e-2 + 1.6e-2 * np.abs(mat)
    max_err = float(np.max(np.abs(fused - mat)))
    assert np.all(np.abs(fused - mat) <= atol), (
        f"Fused NVFP4 matmul mismatch vs materialize oracle: max_err={max_err}"
    )


def _snap_to_e2m1_graph(v: TensorValue) -> TensorValue:
    """Round a graph tensor onto the signed E2M1 grid {0,.5,1,1.5,2,3,4,6}."""
    a = ops.abs(v)
    z = a * 0.0
    sign = ops.where(v < 0.0, z - 1.0, z + 1.0)
    mag = ops.where(
        a > 5.0, z + 6.0,
        ops.where(a >= 3.5, z + 4.0,
        ops.where(a >= 2.5, z + 3.0,
        ops.where(a >= 1.75, z + 2.0,
        ops.where(a >= 1.25, z + 1.5,
        ops.where(a >= 0.75, z + 1.0,
        ops.where(a >= 0.25, z + 0.5, z)))))),
    )
    return sign * mag


def _fake_quant_fp4_act_graph(x: TensorValue) -> TensorValue:
    """In-graph NVFP4 activation fake-quant (bf16 -> fp4 -> bf16), input_scale=1.

    Matches the W4A4 kernel's dynamic per-block-16 activation quantization exactly
    (amax/6 -> fp8 block scale, e2m1 snap), so a materialize-op matmul on this
    tensor is the block-scaled result the native FP4 kernel must reproduce.
    """
    m = int(x.shape[0])
    k = int(x.shape[1])
    n_blk = k // _SF_VECTOR_SIZE
    xf = ops.cast(x, DType.float32)
    xb = ops.reshape(xf, [m, n_blk, _SF_VECTOR_SIZE])
    amax = ops.max(ops.abs(xb), axis=-1)
    scale_q = ops.cast(
        ops.cast(amax / 6.0, DType.float8_e4m3fn), DType.float32
    )
    scale_safe = ops.where(scale_q > 0.0, scale_q, scale_q * 0.0 + 1.0)
    q = _snap_to_e2m1_graph(xb / scale_safe)
    x_hat = ops.reshape(q * scale_q, [m, k])
    return ops.cast(x_hat, DType.bfloat16)


def test_linear_nvfp4_cuda_w4a4_matches_sim() -> None:
    """Native FP4 W4A4 op == fake-quant(activation) + materialize W4A16 oracle.

    The native FP4xFP4 kernel (``mo.matmul.block.scaled.cuda.w4a4``) quantizes the
    activation to fp4 internally and runs the sm_120a block-scaled FP4 tensor-core
    MMA. The reference fake-quantizes the activation with the *identical* recipe
    in-graph, then runs the proven materialize->dense W4A16 oracle on it -- so both
    compute the same block-scaled matmul over the same quantized operands and must
    agree to bf16-MMA tolerance. This is the image-validated W4A4 simulation, so a
    pass means the native kernel reproduces the validated numerics.
    """
    _skip_if_not_cuda_w4a16()

    rng = np.random.default_rng(2)
    m, n, k = 8, 256, 512

    device = Accelerator(0)
    device_ref = DeviceRef(device.label, device.id)

    nibbles = rng.integers(0, 16, size=(n, k), dtype=np.uint8)
    packed = _pack_fp4_weight(nibbles)
    scale_k = k // _SF_VECTOR_SIZE
    scales_fp32 = rng.integers(1, 5, size=(n, scale_k)).astype(
        np.float32
    ) * np.float32(0.5)
    scales_fp8_bytes = _fp32_to_fp8_bytes(scales_fp32, device, device_ref)

    # Wider activation range so the fp4 activation quant is exercised.
    x_fp32 = (rng.standard_normal((m, k)) * 0.5).astype(np.float32)

    session = InferenceSession(devices=[device])
    with Graph(
        "NVFP4_CUDA_W4A4_vs_sim",
        input_types=[
            TensorType(DType.float32, (m, k), device=device_ref),
            TensorType(DType.uint8, (n, k // 2), device=device_ref),
            TensorType(DType.float8_e4m3fn, (n, scale_k), device=device_ref),
        ],
    ) as graph:
        x_in, packed_in, scales_in = graph.inputs
        assert isinstance(x_in, TensorValue)
        assert isinstance(packed_in, TensorValue)
        assert isinstance(scales_in, TensorValue)
        x_bf16 = ops.cast(x_in, DType.bfloat16)

        # Reference: fake-quant the activation to fp4 (identical recipe), then the
        # proven materialize oracle (bf16 MMA over the dequantized operands).
        x_fake = _fake_quant_fp4_act_graph(x_bf16)
        out_sim = _cuda_weight_only_block_scaled_matmul(
            x_fake, packed_in, scales_in, out_type=DType.bfloat16
        )
        # Under test: the native FP4xFP4 W4A4 op. weight_scale_2=1.0 keeps the
        # (now in-kernel) epilogue fold numerically inert for the sim compare.
        ws2_one = ops.constant(
            1.0, dtype=DType.float32, device=DeviceRef.CPU()
        )
        out_w4a4 = _cuda_w4a4_matmul(
            x_bf16, packed_in, scales_in, ws2_one, out_type=DType.bfloat16
        )

        graph.output(
            ops.cast(out_sim, DType.float32),
            ops.cast(out_w4a4, DType.float32),
        )

    compiled = session.load(graph)
    x_dev = Buffer.from_numpy(x_fp32).to(device)
    packed_dev = Buffer.from_numpy(packed).to(device)
    scales_dev = (
        Buffer.from_numpy(scales_fp8_bytes)
        .view(DType.float8_e4m3fn, (n, scale_k))
        .to(device)
    )
    sim_buf, w4a4_buf = compiled.execute(x_dev, packed_dev, scales_dev)

    assert isinstance(sim_buf, Buffer)
    assert isinstance(w4a4_buf, Buffer)
    sim = np.from_dlpack(sim_buf.to(CPU())).astype(np.float32)
    w4a4 = np.from_dlpack(w4a4_buf.to(CPU())).astype(np.float32)

    assert w4a4.shape == (m, n)
    assert np.isfinite(w4a4).all(), "W4A4 NVFP4 output has NaN/Inf"

    atol = 2e-2 + 2e-2 * np.abs(sim)
    max_err = float(np.max(np.abs(w4a4 - sim)))
    assert np.all(np.abs(w4a4 - sim) <= atol), (
        f"Native W4A4 matmul mismatch vs fake-quant sim: max_err={max_err}"
    )

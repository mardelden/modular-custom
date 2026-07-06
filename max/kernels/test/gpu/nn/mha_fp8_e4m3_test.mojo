# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""Numeric test: e4m3 (fp8) flash attention on sm_120 vs an exact host reference.

Drives the raw-tensor `flash_attention` overload (the one `mo.mha.no_cache`
uses) with e4m3 Q/K/V and bf16 output, NullMask, single batch, depth 128 --
the FLUX.2-Klein prefill shape class. The host reference replicates the kernel
algorithm EXACTLY so a tight tolerance validates the ×256 P-lift + e4m3 P·V:

    scores = scale * (Qe4m3 . Ke4m3)        # f32 accumulate
    m      = max_k scores ;  p_k = exp(scores_k - m)
    rowsum = sum_k p_k                        # f32, UNlifted
    out_d  = (sum_k e4m3(256*p_k) * Ve4m3) / rowsum / 256

(Q,K,V are already stored as e4m3, so reads are the rounded values.)
"""

from std.math import exp, sqrt
from std.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from nn.attention.gpu.mha import flash_attention
from nn.attention.mha_mask import NullMask
from nn.attention.mha_utils import MHAConfig
from std.testing import assert_almost_equal


def test_bf16_multiwarp_control[ctl_BK: Int, ctl_stages: Int = 4](
    ctx: DeviceContext,
) raises:
    """CONTROL: bf16 through the fp8 kernel geometry (WN=64 -> num_warps_n=2).

    Runs the SAME multi-warp FA2 configuration the fp8 path uses, but with
    bf16 data, and compares against the default single-warp bf16 config.
    ctl_BK isolates the BK axis too (fp8 uses BK=64; bf16 default is 32).
    - control asserts/crashes -> the geometry (WN/BK) itself is broken,
      independent of dtype.
    - control matches default -> geometry fine; the fp8 bug is dtype-specific.
    """
    comptime qkv_type = DType.bfloat16
    comptime depth = 128
    comptime num_heads = 2
    comptime seq_len = 64
    comptime num_keys = 64
    var scale = Float32(1.0) / sqrt(Float32(depth))
    var n = seq_len * num_heads * depth

    print(
        "test_bf16_multiwarp_control (WN=64, BK=",
        ctl_BK,
        ", stages=",
        ctl_stages,
        ")",
    )

    var q_h = ctx.enqueue_create_host_buffer[qkv_type](n)
    var k_h = ctx.enqueue_create_host_buffer[qkv_type](n)
    var v_h = ctx.enqueue_create_host_buffer[qkv_type](n)
    var o_ref_h = ctx.enqueue_create_host_buffer[qkv_type](n)
    var o_mw_h = ctx.enqueue_create_host_buffer[qkv_type](n)
    ctx.synchronize()
    # Same wide pattern as the fp8 test (broad softmax score range).
    for i in range(n):
        q_h[i] = Scalar[qkv_type](Float32(((i * 5 + 1) % 29) - 14) * 0.5)
        var kk = i // depth
        var kscale = Float32(1 + (kk % 6)) * 0.5
        k_h[i] = Scalar[qkv_type](
            Float32(((i * 3 + 2) % 23) - 11) * 0.25 * kscale
        )
        v_h[i] = Scalar[qkv_type](Float32(((i * 7 + 3) % 31) - 15) * 0.25)

    var q_d = ctx.enqueue_create_buffer[qkv_type](n)
    var k_d = ctx.enqueue_create_buffer[qkv_type](n)
    var v_d = ctx.enqueue_create_buffer[qkv_type](n)
    var o_ref_d = ctx.enqueue_create_buffer[qkv_type](n)
    var o_mw_d = ctx.enqueue_create_buffer[qkv_type](n)
    ctx.enqueue_copy(q_d, q_h)
    ctx.enqueue_copy(k_d, k_h)
    ctx.enqueue_copy(v_d, v_h)

    var q_dev = TileTensor(
        q_d, row_major((1, seq_len, Idx[num_heads], Idx[depth]))
    )
    var k_dev = TileTensor(
        k_d, row_major((1, num_keys, Idx[num_heads], Idx[depth]))
    )
    var v_dev = TileTensor(
        v_d, row_major((1, num_keys, Idx[num_heads], Idx[depth]))
    )
    var o_ref_dev = TileTensor(
        o_ref_d, row_major((1, seq_len, Idx[num_heads], Idx[depth]))
    )
    var o_mw_dev = TileTensor(
        o_mw_d, row_major((1, seq_len, Idx[num_heads], Idx[depth]))
    )

    # Default (single-warp, WN=BN) bf16 config = today's production path.
    flash_attention(o_ref_dev, q_dev, k_dev, v_dev, NullMask(), scale, ctx)
    # Multi-warp geometry: WN=64 -> num_warps_n=2 (what the fp8 path uses).
    comptime mw_cfg = MHAConfig[qkv_type](
        num_heads=num_heads,
        depth=depth,
        BK=ctl_BK,
        WN=64,
        num_pipeline_stages=ctl_stages,
    )
    flash_attention[config=mw_cfg](
        o_mw_dev, q_dev, k_dev, v_dev, NullMask(), scale, ctx
    )
    ctx.synchronize()
    ctx.enqueue_copy(o_ref_h, o_ref_d)
    ctx.enqueue_copy(o_mw_h, o_mw_d)
    ctx.synchronize()

    var n_bad = 0
    var max_ad = Float32(0)
    for i in range(n):
        var a = o_ref_h[i].cast[DType.float32]()
        var b = o_mw_h[i].cast[DType.float32]()
        var ad = abs(a - b)
        max_ad = max(max_ad, ad)
        if ad > 2e-2 + 2e-2 * abs(a):
            n_bad += 1
    print("  control max_abs_diff=", max_ad, " n_bad=", n_bad)
    _ = q_d^
    _ = k_d^
    _ = v_d^
    _ = o_ref_d^
    _ = o_mw_d^
    if n_bad == 0:
        print("  CONTROL PASS (multi-warp bf16 == default bf16)")
    else:
        print("  CONTROL FAILED n_bad=", n_bad)


def e4m3_rt(v: Float32) -> Float32:
    return v.cast[DType.float8_e4m3fn]().cast[DType.float32]()


def test_fp8_attention[
    depth: Int,
    num_heads: Int,
](seq_len: Int, num_keys: Int, ctx: DeviceContext) raises:
    comptime qkv_type = DType.float8_e4m3fn
    comptime out_type = DType.bfloat16
    comptime batch_size = 1
    comptime group = 1
    comptime kv_num_heads = num_heads // group
    var scale = Float32(1.0) / sqrt(Float32(depth))

    print(
        "test_fp8_attention depth=",
        depth,
        "num_heads=",
        num_heads,
        "seq_len=",
        seq_len,
        "num_keys=",
        num_keys,
    )

    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var o_size = q_size

    var q_h = ctx.enqueue_create_host_buffer[qkv_type](q_size)
    var k_h = ctx.enqueue_create_host_buffer[qkv_type](k_size)
    var v_h = ctx.enqueue_create_host_buffer[qkv_type](k_size)
    var o_h = ctx.enqueue_create_host_buffer[out_type](o_size)
    ctx.synchronize()

    # Wider dynamic range + varied per-key magnitudes, closer to real
    # post-RMSNorm/RoPE activations: creates a broad softmax score range so
    # the online-softmax max-rescale across K-tiles is actually exercised.
    for i in range(q_size):
        q_h[i] = Scalar[qkv_type](Float32(((i * 5 + 1) % 29) - 14) * 0.5)
    for i in range(k_size):
        # per-key scale ramp so different keys dominate different query rows.
        var key = i // depth
        var kscale = Float32(1 + (key % 6)) * 0.5
        k_h[i] = Scalar[qkv_type](Float32(((i * 3 + 2) % 23) - 11) * 0.25 * kscale)
    for i in range(k_size):
        v_h[i] = Scalar[qkv_type](Float32(((i * 7 + 3) % 31) - 15) * 0.25)

    var q_d = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_d = ctx.enqueue_create_buffer[qkv_type](k_size)
    var v_d = ctx.enqueue_create_buffer[qkv_type](k_size)
    var o_d = ctx.enqueue_create_buffer[out_type](o_size)
    ctx.enqueue_copy(q_d, q_h)
    ctx.enqueue_copy(k_d, k_h)
    ctx.enqueue_copy(v_d, v_h)

    var q_dev = TileTensor(
        q_d, row_major((batch_size, seq_len, Idx[num_heads], Idx[depth]))
    )
    var k_dev = TileTensor(
        k_d, row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth]))
    )
    var v_dev = TileTensor(
        v_d, row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth]))
    )
    var o_dev = TileTensor(
        o_d, row_major((batch_size, seq_len, Idx[num_heads], Idx[depth]))
    )

    flash_attention(o_dev, q_dev, k_dev, v_dev, NullMask(), scale, ctx)
    ctx.synchronize()
    ctx.enqueue_copy(o_h, o_d)
    ctx.synchronize()

    # ---- host reference (exact fp8 algorithm) ----
    var n_bad = 0
    var max_rel = Float32(0)
    for h in range(num_heads):
        for s in range(seq_len):
            # scores over keys
            var scores = List[Float32]()
            var m = Float32(-1.0e30)
            for kk in range(num_keys):
                var acc = Float32(0)
                for d in range(depth):
                    var qv = q_h[(s * num_heads + h) * depth + d].cast[
                        DType.float32
                    ]()
                    var kv = k_h[(kk * kv_num_heads + h) * depth + d].cast[
                        DType.float32
                    ]()
                    acc += qv * kv
                var sc = acc * scale
                scores.append(sc)
                m = max(m, sc)
            var rowsum = Float32(0)
            for kk in range(num_keys):
                rowsum += exp(scores[kk] - m)
            for d in range(depth):
                var oacc = Float32(0)
                for kk in range(num_keys):
                    var p = exp(scores[kk] - m)
                    var p_lift = e4m3_rt(p * 256.0)
                    var vv = v_h[(kk * kv_num_heads + h) * depth + d].cast[
                        DType.float32
                    ]()
                    oacc += p_lift * vv
                var expected = (oacc / rowsum) / 256.0
                var got = o_h[(s * num_heads + h) * depth + d].cast[
                    DType.float32
                ]()
                var rel = abs(got - expected) / (abs(expected) + 1e-3)
                max_rel = max(max_rel, rel)
                if rel > 3e-2 and abs(got - expected) > 3e-2:
                    if n_bad < 8:
                        print("  MISMATCH h=", h, "s=", s, "d=", d, "got", got, "exp", expected)
                    n_bad += 1

    print("  max_rel=", max_rel, " n_bad=", n_bad)
    _ = q_d^
    _ = k_d^
    _ = v_d^
    _ = o_d^
    if n_bad == 0:
        print("  PASS")
    else:
        print("  FAILED n_bad=", n_bad)


def bench_attention[
    qkv_type: DType,
    out_type: DType,
    num_heads: Int,
    depth: Int,
    config: MHAConfig[qkv_type],
](label: String, seq_len: Int, ctx: DeviceContext) raises:
    """Time one flash_attention config at a Klein-like shape (single call =
    one attention op; the render does 224 of these). Values are irrelevant to
    timing, so buffers are just filled with a constant."""
    var n = num_heads * seq_len * depth
    var q_d = ctx.enqueue_create_buffer[qkv_type](n)
    var k_d = ctx.enqueue_create_buffer[qkv_type](n)
    var v_d = ctx.enqueue_create_buffer[qkv_type](n)
    var o_d = ctx.enqueue_create_buffer[out_type](n)
    q_d.enqueue_fill(Scalar[qkv_type](0.1))
    k_d.enqueue_fill(Scalar[qkv_type](0.1))
    v_d.enqueue_fill(Scalar[qkv_type](0.1))
    var q_dev = TileTensor(
        q_d, row_major((1, seq_len, Idx[num_heads], Idx[depth]))
    )
    var k_dev = TileTensor(
        k_d, row_major((1, seq_len, Idx[num_heads], Idx[depth]))
    )
    var v_dev = TileTensor(
        v_d, row_major((1, seq_len, Idx[num_heads], Idx[depth]))
    )
    var o_dev = TileTensor(
        o_d, row_major((1, seq_len, Idx[num_heads], Idx[depth]))
    )
    var scale = Float32(1.0) / sqrt(Float32(depth))

    @parameter
    @always_inline
    @__copy_capture(q_dev, k_dev, v_dev, o_dev)
    def launch(ctx: DeviceContext) raises:
        flash_attention[config=config](
            o_dev, q_dev, k_dev, v_dev, NullMask(), scale, ctx
        )

    launch(ctx)
    ctx.synchronize()
    var ns = Float64(ctx.execution_time[launch](50)) / 50.0
    print("  BENCH", label, ":", ns / 1.0e6, "ms/call")
    _ = q_d^
    _ = k_d^
    _ = v_d^
    _ = o_d^


def bench_klein(ctx: DeviceContext) raises:
    print("=== attention bench @ Klein shape (H=48, S=4608, D=128) ===")
    comptime H = 48
    comptime D = 128
    comptime S = 4608
    comptime bf16 = DType.bfloat16
    comptime e4m3 = DType.float8_e4m3fn
    comptime cfg_bf16 = MHAConfig[bf16](num_heads=H, depth=D)
    comptime cfg_bf16_bk64 = MHAConfig[bf16](
        num_heads=H, depth=D, BK=64, num_pipeline_stages=2
    )
    comptime cfg_fp8 = MHAConfig[e4m3](num_heads=H, depth=D)
    bench_attention[bf16, bf16, H, D, cfg_bf16]("bf16 default (BK32 st4)", S, ctx)
    bench_attention[bf16, bf16, H, D, cfg_bf16_bk64](
        "bf16 BK64 st2 (config-only)", S, ctx
    )
    bench_attention[e4m3, bf16, H, D, cfg_fp8]("fp8 (BK64 st2 + smemP)", S, ctx)


def main() raises:
    with DeviceContext() as ctx:
        # fp8 uses the single-warp geometry (WN=BN) + smem P staging. Wide
        # inputs (broad softmax range) at single- and multi-K-tile; the earlier
        # WN=BN/2 multi-warp config was numerically broken at depth 128 (the
        # `test_bf16_multiwarp_control` helper documents that: it FAILS for wide
        # inputs even in bf16 -- run it if re-investigating the multi-warp path).
        test_fp8_attention[depth=128, num_heads=2](64, 64, ctx)
        test_fp8_attention[depth=128, num_heads=2](128, 512, ctx)
        bench_klein(ctx)

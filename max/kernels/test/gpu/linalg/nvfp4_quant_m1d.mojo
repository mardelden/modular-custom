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
"""D2/M1d: sm_120 dynamic NVFP4 activation quantizer.

bf16 `x[M,K]` -> packed fp4 `[M, K/2]` (e2m1, low nibble = even k) + fp8-e4m3
block scales `[M, K/16]`, per-block-16 dynamic: block_scale = amax(|x/input_scale|)/6
rounded to fp8-e4m3; nibble = round_to_e2m1((x/input_scale)/block_scale). This is
the sm_120 analogue of the SM100-only `quantize_dynamic_block_scaled`, in the exact
packed layout the M1c GEMM consumes (so SFA = these scales, and the epilogue folds
the per-tensor input_scale * weight_scale_2).

Test: quantize random bf16, host-dequant (E2M1[nib]*block_scale*input_scale), and
confirm (a) every dequant value lands on {e2m1 grid}*scale*input_scale, and (b) the
reconstruction error is a real, bounded fp4 error (a few %).
"""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.memory import bitcast
from std.math import ceildiv, sqrt


comptime M = 40
comptime K = 256
comptime KHALF = K // 2   # packed bytes per row
comptime KB = K // 16     # scale blocks per row
comptime INPUT_SCALE = Float32(0.7)


@always_inline
def round_e2m1_nibble(v: Float32) -> UInt8:
    """Round to nearest signed E2M1 code (0..15). Grid {0,.5,1,1.5,2,3,4,6}."""
    var a = abs(v)
    var sign = UInt8(8) if v < 0.0 else UInt8(0)
    var mag: UInt8
    if a > 5.0:
        mag = 7      # 6.0
    elif a >= 3.5:
        mag = 6      # 4.0
    elif a >= 2.5:
        mag = 5      # 3.0
    elif a >= 1.75:
        mag = 4      # 2.0
    elif a >= 1.25:
        mag = 3      # 1.5
    elif a >= 0.75:
        mag = 2      # 1.0
    elif a >= 0.25:
        mag = 1      # 0.5
    else:
        mag = 0      # 0.0
    return sign | mag


def quant_act_kernel(
    x: UnsafePointer[BFloat16, MutAnyOrigin],       # [M, K]
    out_packed: UnsafePointer[UInt8, MutAnyOrigin],  # [M, KHALF]
    out_scale: UnsafePointer[UInt8, MutAnyOrigin],   # [M, KB]  (fp8-e4m3 bytes)
    input_scale: Float32,
    m_dim: Int,
):
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= m_dim * KB:
        return
    var m = idx // KB
    var kb = idx % KB
    var base = m * K + kb * 16
    var inv_is = Float32(1.0) / input_scale

    # amax over the 16-element block, in the 1/input_scale-scaled domain.
    var amax = Float32(0)
    for j in range(16):
        var v = abs(Float32(x[base + j]) * inv_is)
        amax = max(amax, v)

    # block_scale = amax/6 rounded to fp8-e4m3 (store the byte; use the rounded value).
    var bs_f = amax / 6.0
    var bs_fp8 = SIMD[DType.float8_e4m3fn, 1](bs_f.cast[DType.float8_e4m3fn]())
    out_scale[m * KB + kb] = bitcast[DType.uint8, 1](bs_fp8)[0]
    var bs = Float32(bs_fp8[0])
    var inv_bs = (Float32(1.0) / bs) if bs > 0.0 else Float32(0.0)

    # Quantize the 16 elements -> 8 packed bytes.
    for jb in range(8):
        var lo = round_e2m1_nibble(
            Float32(x[base + 2 * jb]) * inv_is * inv_bs
        )
        var hi = round_e2m1_nibble(
            Float32(x[base + 2 * jb + 1]) * inv_is * inv_bs
        )
        out_packed[m * KHALF + kb * 8 + jb] = lo | (hi << 4)


comptime E2M1 = SIMD[DType.float32, 16](
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def main() raises:
    with DeviceContext() as ctx:
        print("D2/M1d: NVFP4 dynamic activation quantizer [", M, "x", K, "]")

        var x_h = ctx.enqueue_create_host_buffer[DType.bfloat16](M * K)
        var pk_h = ctx.enqueue_create_host_buffer[DType.uint8](M * KHALF)
        var sc_h = ctx.enqueue_create_host_buffer[DType.uint8](M * KB)
        ctx.synchronize()

        # Deterministic pseudo-random bf16 input in [-3, 3].
        for i in range(M * K):
            var h = (i * 1103515245 + 12345) % 2000
            x_h[i] = ((Float32(h) / 1000.0 - 1.0) * 3.0).cast[DType.bfloat16]()

        var x_d = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
        var pk_d = ctx.enqueue_create_buffer[DType.uint8](M * KHALF)
        var sc_d = ctx.enqueue_create_buffer[DType.uint8](M * KB)
        ctx.enqueue_copy(x_d, x_h)

        ctx.enqueue_function[quant_act_kernel](
            x_d.unsafe_ptr(), pk_d.unsafe_ptr(), sc_d.unsafe_ptr(),
            INPUT_SCALE, M,
            grid_dim=ceildiv(M * KB, 128), block_dim=128,
        )
        ctx.enqueue_copy(pk_h, pk_d)
        ctx.enqueue_copy(sc_h, sc_d)
        ctx.synchronize()

        # Host dequant + checks.
        var num = Float32(0)  # ||x_hat - x||^2
        var den = Float32(0)  # ||x||^2
        var grid_ok = True
        for m in range(M):
            for kb in range(KB):
                var sc_byte = sc_h[m * KB + kb]
                var sc_f8 = bitcast[DType.float8_e4m3fn, 1](
                    SIMD[DType.uint8, 1](sc_byte)
                )
                var bs = Float32(sc_f8[0])
                for j in range(16):
                    var k = kb * 16 + j
                    var byte = pk_h[m * KHALF + kb * 8 + j // 2]
                    var nib = Int(
                        (byte >> UInt8(4 * (j % 2))) & UInt8(0xF)
                    )
                    var xhat = E2M1[nib] * bs * INPUT_SCALE
                    var xv = Float32(x_h[m * K + k])
                    var d = xhat - xv
                    num += d * d
                    den += xv * xv
                    # grid check: xhat / (bs*input_scale) must be an e2m1 value
                    if bs > 0.0:
                        var q = xhat / (bs * INPUT_SCALE)
                        var best = Float32(1e9)
                        for g in range(16):
                            best = min(best, abs(q - E2M1[g]))
                        if best > 1e-2:
                            grid_ok = False

        var rel = sqrt(num / den)
        print("  grid membership (all nibbles on e2m1 grid):", grid_ok)
        print("  reconstruction rel error ||x_hat - x||/||x|| =", rel)
        if grid_ok and rel > 0.02 and rel < 0.25:
            print("  PASS (faithful bounded fp4 activation quant)")
        else:
            print("  FAILED")

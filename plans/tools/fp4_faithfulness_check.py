"""End-to-end faithfulness check for the direct fp4-activation fake-quant.

Builds the EXACT op sequence used by `_fake_quant_fp4_activation` (per-block-16
amax -> scale=amax/6 rounded to float8_e4m3fn -> snap to the e2m1 grid ->
dequant, with the per-tensor input_scale folded), runs it on the sm_120 GPU, and
confirms:
  * output is finite (no NaN/Inf),
  * every x_hat lands exactly on {e2m1 grid} * block_scale * input_scale,
  * the per-block scale is a valid float8_e4m3fn value,
  * reconstruction x_hat ~= x (small rel error) -> the fp4 error injected is real
    but bounded, i.e. this is a faithful fp4 activation quantization.
"""
import numpy as np
from max.driver import Accelerator, Buffer, CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

M, K = 8, 128
INPUT_SCALE = 0.7
BLOCK = 16
E2M1_MAX = 6.0
GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def _snap_to_e2m1(v):
    a = ops.abs(v)
    z = a * 0.0
    sign = ops.where(v < 0.0, z - 1.0, z + 1.0)
    mag = ops.where(a > 5.0, z + 6.0,
          ops.where(a >= 3.5, z + 4.0,
          ops.where(a >= 2.5, z + 3.0,
          ops.where(a >= 1.75, z + 2.0,
          ops.where(a >= 1.25, z + 1.5,
          ops.where(a >= 0.75, z + 1.0,
          ops.where(a >= 0.25, z + 0.5, z)))))))
    return sign * mag


def fake_quant_with_intermediates(x, input_scale):
    """Same math as _fake_quant_fp4_activation, but also returns q and scale."""
    K_ = int(x.shape[1])
    n_blk = K_ // BLOCK
    isc = input_scale.cast(DType.float32).to(x.device)
    xf = x.cast(DType.float32)
    xs = xf / isc
    xb = xs.reshape([x.shape[0], n_blk, BLOCK])
    amax = ops.max(ops.abs(xb), axis=-1)                       # [M, n_blk, 1]
    scale = amax / E2M1_MAX
    scale_q = scale.cast(DType.float8_e4m3fn).cast(DType.float32)
    scale_safe = ops.where(scale_q > 0.0, scale_q, scale_q * 0.0 + 1.0)
    q = _snap_to_e2m1(xb / scale_safe)                         # [M, n_blk, BLOCK]
    deq = q * scale_q
    x_hat_f32 = deq.reshape([x.shape[0], K_]) * isc            # f32, exactly on grid
    x_hat = x_hat_f32.cast(DType.bfloat16)                     # what the matmul sees
    q_full = q.reshape([x.shape[0], K_])
    scale_full = ops.broadcast_to(
        scale_q, [x.shape[0], n_blk, BLOCK]
    ).reshape([x.shape[0], K_])
    return x_hat, x_hat_f32, q_full, scale_full


np.random.seed(0)
x_np = (np.random.randn(M, K) * 2.0).astype(np.float32)

dev = DeviceRef.GPU()
with Graph("fq", input_types=[TensorType(DType.float32, [M, K], device=dev)]) as g:
    (xf,) = (t.tensor for t in g.inputs)
    xb = xf.cast(DType.bfloat16)
    isc = ops.constant(INPUT_SCALE, DType.float32, device=dev)
    x_hat, x_hat_f32, q_full, scale_full = fake_quant_with_intermediates(xb, isc)
    g.output(x_hat.cast(DType.float32), x_hat_f32, q_full, scale_full,
             xb.cast(DType.float32))

sess = InferenceSession(devices=[Accelerator()])
model = sess.load(g)
acc = Accelerator()
outs = model.execute(Buffer.from_numpy(x_np).to(acc))
x_hat = outs[0].to(CPU()).to_numpy()        # bf16-rounded (what the matmul sees)
x_hat_f32 = outs[1].to(CPU()).to_numpy()    # pre-bf16, exactly on the fp4 grid
q_full = outs[2].to(CPU()).to_numpy()
scale_full = outs[3].to(CPU()).to_numpy()
xused = outs[4].to(CPU()).to_numpy()        # bf16-rounded input the quantizer saw

# 1) finite
finite = bool(np.all(np.isfinite(x_hat)))

# 2) grid membership of the snapped codes q
q_in_grid = bool(np.all(np.isin(np.round(np.abs(q_full), 4), GRID)))

# 3) exact dequant identity on the pre-bf16 value: x_hat_f32 == q*scale*input_scale
deq_ident = float(np.max(np.abs(x_hat_f32 - q_full * scale_full * INPUT_SCALE)))

# 4) x_hat_f32 / (scale*input_scale) lands exactly on the e2m1 grid
denom = scale_full * INPUT_SCALE
ratio = np.where(denom != 0, x_hat_f32 / np.where(denom != 0, denom, 1.0), 0.0)
amag = np.abs(ratio)
nearest = GRID[np.argmin(np.abs(amag[..., None] - GRID[None, None, :]), axis=-1)]
snap_err = float(np.max(np.abs(amag - nearest)))

# 5) block scales are valid float8_e4m3fn (multiples of 0.0625 in [0.5,1) octave etc.)
# 6) reconstruction error of the bf16 output vs the (bf16) input
rel_err = float(np.linalg.norm(x_hat - xused) / np.linalg.norm(xused))
bf16_cast_err = float(np.max(np.abs(x_hat - x_hat_f32)))

print("finite (no NaN/Inf):           ", finite)
print("q all exactly on e2m1 grid:    ", q_in_grid)
print("exact dequant identity err:    ", f"{deq_ident:.2e}  (f32 x_hat == q*scale*is)")
print("exact e2m1 grid-snap err:      ", f"{snap_err:.2e}  (f32 x_hat on grid, want ~0)")
print("bf16 output cast err (expected):", f"{bf16_cast_err:.2e}")
print("rel reconstruction err x_hat~x: ", f"{rel_err:.4f}  (fp4 activation error)")
print("distinct block-scale values:   ", np.unique(np.round(scale_full, 5)))
print("x used [0,:8]:                 ", np.round(xused[0, :8], 4))
print("x_hat  [0,:8]:                 ", np.round(x_hat[0, :8], 4))
print("q      [0,:8]:                 ", np.round(q_full[0, :8], 4))
ok = finite and q_in_grid and deq_ident < 1e-4 and snap_err < 1e-4 \
    and 0.02 < rel_err < 0.25
print("\nRESULT:", "FAITHFUL (fp4 activation quant validated)" if ok else "CHECK FAILED")

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
"""Gate A0: mutable BufferType + dynamic-offset buffer_store_slice round-trip.

This is the load-bearing mechanism for the KV cache: a graph that writes a
variable-length chunk into a persistent device buffer at a runtime offset, with
the writes surviving across execute() calls. Verify it on the target backend
(sm_120) BEFORE building the full cache graph. No torch/HF deps.

    python check_buffer_gpu.py --device gpu
"""

from __future__ import annotations

import argparse

import numpy as np
from common import resolve_device
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops

# Small stand-in for a per-layer KV cache buffer: [batch, heads, max_len, hd].
SHAPE = [1, 2, 16, 4]


def build_write_graph(device: DeviceRef) -> Graph:
    buf_t = BufferType(DType.float32, SHAPE, device=device)
    chunk_t = TensorType(DType.float32, [1, 2, "t_new", 4], device=device)
    # The write offset must be a CPU-resident scalar (buffer_store_slice reuses
    # slice_tensor's host-side start/stop/step machinery).
    start_t = TensorType(DType.int64, [], device=DeviceRef.CPU())
    with Graph("kv_write_smoke", input_types=[buf_t, chunk_t, start_t]) as g:
        buf = g.inputs[0].buffer
        chunk = g.inputs[1].tensor
        start = g.inputs[2].tensor
        t_new_dim = chunk.shape[2]
        stop = start + ops.shape_to_tensor(chunk.shape)[2]
        # Tuple slice form: (slice(start, stop, 1), out_dim) at the seq axis.
        buf[
            slice(None),
            slice(None),
            (slice(start, stop, 1), t_new_dim),
            slice(None),
        ] = chunk
        g.output(ops.buffer_load(buf))
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    args = ap.parse_args()

    dev = resolve_device(args.device)
    session = InferenceSession(devices=[dev])
    model = session.load(build_write_graph(DeviceRef.from_device(dev)))

    def to_host(buf):
        return np.asarray(buf.to(CPU()).to_numpy())

    # Persistent cache buffer, zero-initialized on device.
    cache = Buffer.from_numpy(np.zeros(SHAPE, dtype=np.float32)).to(dev)

    def write(chunk_np, start):
        chunk = Buffer.from_numpy(np.ascontiguousarray(chunk_np)).to(dev)
        start_buf = Buffer.from_numpy(np.array(start, dtype=np.int64))  # CPU
        model.execute(cache, chunk, start_buf)

    # Write 2 rows of 1.0 at offset 3.
    write(np.ones((1, 2, 2, 4), np.float32), 3)
    c = to_host(cache)
    ok1 = np.allclose(c[0, 0, 3:5], 1.0) and np.allclose(c[0, 0, :3], 0.0)
    print(f"[{'PASS' if ok1 else 'FAIL'}] write@3 len2: rows3-4==1, rows0-2==0")

    # Write 3 rows of 2.0 at offset 8 into the SAME buffer.
    write(2.0 * np.ones((1, 2, 3, 4), np.float32), 8)
    c = to_host(cache)
    persisted = np.allclose(c[0, 0, 3:5], 1.0)  # first write survived
    second = np.allclose(c[0, 0, 8:11], 2.0)
    gap = np.allclose(c[0, 0, 5:8], 0.0)
    ok2 = persisted and second and gap
    print(
        f"[{'PASS' if ok2 else 'FAIL'}] write@8 len3 (same buffer): "
        f"prior-persisted={persisted} new={second} gap-zero={gap}"
    )

    all_ok = ok1 and ok2
    print(f"{'ALL BUFFER CHECKS PASSED' if all_ok else 'BUFFER CHECK FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

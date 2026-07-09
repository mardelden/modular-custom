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

"""Spatial tiled VAE decode for high-resolution images.

Splits a spatial NCHW latent into overlapping H/W tiles, decodes each tile
through the *same* (symbolic-shape) decoder, and feather-blends the overlapping
output regions. This enables 4K+ output that would otherwise OOM the
full-resolution conv/GroupNorm intermediates -- e.g. a 4096^2 image unpacks to a
``[1,16,512,512]`` latent whose first conv intermediate alone needs ~47 GB.

The helper is decoder-agnostic: it only needs a ``decode_fn`` (latent tile ->
image tile) and the decoder's spatial ``upsample`` factor. Each ``decode_fn``
call is a separate compiled graph, so tiles are decoded one at a time and their
conv intermediates freed before the next -- that bounds peak GPU memory.

Blending is done on the HOST: the compiled decoder reuses its output buffer
across calls, so each decoded tile is copied to a numpy array immediately (this
captures its content before the next decode overwrites the buffer), accumulated
with a separable raised-cosine (Hann) feather, and the final image is uploaded
once. Accumulate-and-divide (``sum(w*tile)/sum(w)``) is correct for arbitrary
overlap and keeps full weight at the true image border (no darkened frame). The
host arrays are cheap (a 4K image is ~800 MB f32) and the decode stays on GPU.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor


def _tile_starts(extent: int, tile: int, overlap: int) -> list[int]:
    """Evenly-spaced tile origins covering ``[0, extent)``.

    Tiles are spaced so the first starts at 0, the last ends exactly at
    ``extent``, and adjacent tiles overlap by at least ``overlap`` (even spacing
    may give slightly more, which the blend handles). Returns ``[0]`` when a
    single tile already covers the extent.
    """
    if tile >= extent:
        return [0]
    step = max(1, tile - overlap)
    n = max(2, -(-(extent - overlap) // step))  # ceil((extent-overlap)/step)
    starts = [round(i * (extent - tile) / (n - 1)) for i in range(n)]
    return sorted(set(starts))


def _ramp_1d(
    length: int, ramp: int, taper_lo: bool, taper_hi: bool
) -> np.ndarray:
    """1-D blend weight: raised-cosine 0->1 over ``ramp`` px on tapered ends."""
    w = np.ones(length, dtype=np.float32)
    if ramp > 0:
        r = min(ramp, length)
        rc = 0.5 * (1.0 - np.cos(np.pi * (np.arange(r) + 0.5) / r))
        if taper_lo:
            w[:r] = rc
        if taper_hi:
            w[-r:] = rc[::-1]
    return w


def tiled_decode(
    decode_fn: Callable[[Tensor], Tensor],
    latent: Tensor,
    *,
    tile_latent: int,
    overlap_latent: int,
    upsample: int = 8,
) -> Tensor:
    """Decode a spatial NCHW latent in overlapping tiles and feather-blend.

    Args:
        decode_fn: Maps a latent tile ``[B, C, h, w]`` to an image
            ``[B, 3, upsample*h, upsample*w]`` (e.g. ``vae.decode``).
        latent: Spatial NCHW latent ``[B, C, H, W]`` (post-denorm).
        tile_latent: Tile edge in latent pixels.
        overlap_latent: Minimum overlap band in latent pixels (< ``tile_latent``).
        upsample: Decoder spatial upsample factor (e.g. 8).

    Returns:
        Decoded image ``[B, 3, upsample*H, upsample*W]`` (device tensor).
    """
    height = int(latent.shape[2])
    width = int(latent.shape[3])
    tile = max(1, int(tile_latent))
    overlap = max(0, min(int(overlap_latent), tile - 1))

    # One tile already covers it -> plain decode (byte-identical to untiled).
    if height <= tile and width <= tile:
        return decode_fn(latent)

    u = int(upsample)
    ys = _tile_starts(height, tile, overlap)
    xs = _tile_starts(width, tile, overlap)
    th = min(tile, height)
    tw = min(tile, width)
    oh, ow = u * th, u * tw
    out_h, out_w = u * height, u * width

    # Feather ramp spans the actual (uniform) overlap so adjacent tapers align.
    step_y = (ys[1] - ys[0]) if len(ys) > 1 else th
    step_x = (xs[1] - xs[0]) if len(xs) > 1 else tw
    ramp_y = u * max(0, th - step_y)
    ramp_x = u * max(0, tw - step_x)

    batch = int(latent.shape[0])
    device = latent.device
    out_sum = np.zeros((batch, 3, out_h, out_w), dtype=np.float32)
    wt_sum = np.zeros((1, 1, out_h, out_w), dtype=np.float32)
    out_dtype = latent.dtype

    for y0 in ys:
        for x0 in xs:
            img_tile = decode_fn(latent[:, :, y0 : y0 + th, x0 : x0 + tw])
            out_dtype = img_tile.dtype
            # Copy to host immediately -- the decoder reuses its output buffer
            # on the next call, so this captures the tile before it's clobbered.
            img_np = np.from_dlpack(img_tile.cast(DType.float32).to(CPU()))
            wy = _ramp_1d(oh, ramp_y, y0 != 0, (y0 + th) != height)
            wx = _ramp_1d(ow, ramp_x, x0 != 0, (x0 + tw) != width)
            win = np.outer(wy, wx).reshape(1, 1, oh, ow)
            iy0, ix0 = u * y0, u * x0
            out_sum[:, :, iy0 : iy0 + oh, ix0 : ix0 + ow] += img_np * win
            wt_sum[:, :, iy0 : iy0 + oh, ix0 : ix0 + ow] += win

    blended = np.ascontiguousarray(out_sum / wt_sum, dtype=np.float32)
    tensor = F.constant(blended, dtype=DType.float32, device=device)
    return tensor.cast(out_dtype)

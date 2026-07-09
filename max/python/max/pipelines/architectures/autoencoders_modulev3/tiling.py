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
image tile) and the decoder's spatial ``upsample`` factor, so it can be reused
across pipelines that share (or differ in) their VAE decoder. It relies on the
fact that each ``decode_fn`` call is a separate compiled graph, so tiles are
materialized and their intermediates freed one at a time -- that is what bounds
peak memory.

2D convs bleed at tile borders, so tiles overlap and are blended with a
separable raised-cosine (Hann) feather. Blending uses accumulate-and-divide
(``sum(weight*tile) / sum(weight)``), which is correct for arbitrary/irregular
overlap and keeps full weight at the true image border (no darkened frame).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor

# Baked separable feather windows, keyed by geometry + taper flags + dtype.
_WINDOW_CACHE: dict[tuple, Tensor] = {}


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


def _window(
    oh: int,
    ow: int,
    ramp_y: int,
    ramp_x: int,
    taper_top: bool,
    taper_bot: bool,
    taper_left: bool,
    taper_right: bool,
    dtype,
    device,
) -> Tensor:
    """Separable feather window ``[1, 1, oh, ow]`` (cached)."""
    key = (
        oh, ow, ramp_y, ramp_x,
        taper_top, taper_bot, taper_left, taper_right,
        dtype, str(device),
    )
    cached = _WINDOW_CACHE.get(key)
    if cached is not None:
        return cached
    wy = _ramp_1d(oh, ramp_y, taper_top, taper_bot)
    wx = _ramp_1d(ow, ramp_x, taper_left, taper_right)
    w2d = np.ascontiguousarray(np.outer(wy, wx).reshape(1, 1, oh, ow))
    # F.constant requires value.dtype == requested dtype, and numpy has no
    # bf16 -> bake as float32, then cast to the decode output dtype.
    tensor = F.constant(w2d, dtype=DType.float32, device=device).cast(dtype)
    _WINDOW_CACHE[key] = tensor
    return tensor


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
            ``[B, 3, upsample*h, upsample*w]`` (e.g. ``vae.decode``). Each call
            is a separate compiled graph, so tiles decode and free their conv
            intermediates one at a time -- this bounds peak memory.
        latent: Spatial NCHW latent ``[B, C, H, W]`` (post-denorm).
        tile_latent: Tile edge in latent pixels.
        overlap_latent: Minimum overlap band in latent pixels (< ``tile_latent``).
        upsample: Decoder spatial upsample factor (e.g. 8).

    Returns:
        Decoded image ``[B, 3, upsample*H, upsample*W]``.
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

    out_sum: Tensor | None = None
    wt_sum: Tensor | None = None
    for y0 in ys:
        for x0 in xs:
            img_tile = decode_fn(latent[:, :, y0 : y0 + th, x0 : x0 + tw])
            win = _window(
                oh,
                ow,
                ramp_y,
                ramp_x,
                taper_top=y0 != 0,
                taper_bot=(y0 + th) != height,
                taper_left=x0 != 0,
                taper_right=(x0 + tw) != width,
                dtype=img_tile.dtype,
                device=img_tile.device,
            )
            iy0, ix0 = u * y0, u * x0
            # F.pad order: [N_before,N_after, C.., H.., W..].
            pads = [
                0, 0,
                0, 0,
                iy0, out_h - iy0 - oh,
                ix0, out_w - ix0 - ow,
            ]
            weighted = F.pad(img_tile * win, pads)  # [B,3,out_h,out_w]
            w_pad = F.pad(win, pads)                # [1,1,out_h,out_w]
            out_sum = weighted if out_sum is None else out_sum + weighted
            wt_sum = w_pad if wt_sum is None else wt_sum + w_pad

    assert out_sum is not None and wt_sum is not None
    # Every output pixel is covered by >= 1 tile with positive weight (border
    # tiles keep full weight), so wt_sum > 0 everywhere -- divide directly.
    return out_sum / wt_sum

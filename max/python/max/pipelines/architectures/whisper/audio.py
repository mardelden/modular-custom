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
"""Audio front-end: load a waveform to 16 kHz mono and extract log-mel features.

Heavy/optional deps (soundfile, scipy) are imported lazily inside functions so
the package's bazel dep-check (which globs ``**/*.py``) doesn't require them.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
HOP_LENGTH = 160  # mel-frame hop; SAMPLE_RATE / HOP_LENGTH = 100 frames/sec
N_FRAMES = 3000  # 30s * 100 frames/sec (the encoder's fixed input length)


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load ``path`` as a mono float32 waveform resampled to ``target_sr``."""
    data: np.ndarray
    sr: int
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        import wave

        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            data = data.reshape(-1, n_channels)

    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != target_sr:
        from math import gcd

        try:
            from scipy.signal import resample_poly

            g = gcd(int(sr), int(target_sr))
            data = resample_poly(data, target_sr // g, sr // g)
        except Exception:
            n_new = int(round(len(data) * target_sr / sr))
            data = np.interp(
                np.linspace(0, len(data), n_new, endpoint=False),
                np.arange(len(data)),
                data,
            )

    return np.ascontiguousarray(data, dtype=np.float32)


def extract_features(
    audio: np.ndarray, feature_extractor
) -> tuple[np.ndarray, int]:
    """Return ``(mel [1, n_mels, 3000] float32, num_content_frames)``.

    ``num_content_frames`` is how many mel frames the real audio covers (before
    the extractor pads to 3000); used to crop trailing silence during alignment.
    """
    num_content_frames = min(len(audio) // HOP_LENGTH, N_FRAMES)
    out = feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="np",
        padding="max_length",
    )
    mel = np.asarray(out["input_features"], dtype=np.float32)
    return mel, num_content_frames

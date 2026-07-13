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
"""Whisper speech-to-text architecture (standalone, word-level timestamps).

This is a standalone transcription pipeline (not registered in the arch
registry / not served via ``max serve``). Entry point is
:class:`~.transcribe.WhisperTranscriber`; run it via ``cli.py``. See
``plans/whisper-word-timestamps.md`` for the design.
"""

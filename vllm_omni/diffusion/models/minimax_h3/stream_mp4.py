# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Encode MiniMax-H3's response MP4 while the video VAE still decodes.

The video API encodes a request's frames into an MP4 only after the worker
returned all of them: the decoded ``uint8`` video crosses the result queue
(a device-to-host copy, a shared-memory copy and the API's copy out), then
the API encodes it. When the API announces, through
``STREAMING_RESPONSE_ENCODING_KEY``, how it would encode the frames, the
output rank instead hands every range of frames the streaming decode
finalizes to one :class:`StreamingVideoBytesEncoder` with exactly those
settings, and returns the MP4 bytes.

The decode itself is unchanged: frames are copied out of the decode's own
``uint8`` buffer after their final write, and the finished video must equal
what was streamed, or the request falls back to returning raw frames. Every
encoder setting comes from the API, and the muxing reproduces the API's
encode byte for byte, so the response MP4 is the one the API would have
produced. ``VLLM_OMNI_H3_STREAM_MP4=0`` restores the raw-frame path.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.utils.chunked_video import ChunkLease, PinnedChunkRing
from vllm_omni.entrypoints.openai.video_api_utils import (
    STREAMING_RESPONSE_ENCODING_KEY,
    STREAMING_RESPONSE_FRAME_PATHS,
    StreamingVideoBytesEncoder,
    usable_cpu_count,
)

logger = init_logger(__name__)

# Transitional switch: unset/1 streams the response MP4 when the API asks for
# it; 0 restores returning the decoded frames for the API to encode.
MINIMAX_H3_STREAM_MP4_ENV = "VLLM_OMNI_H3_STREAM_MP4"
_ENABLED_VALUES = frozenset(("", "1", "true", "on", "yes"))
_DISABLED_VALUES = frozenset(("0", "false", "off", "no"))

# Chunks in flight to the host. One decoded clip is ~17 frames (~53 MB at
# 1344x768); the encoder keeps up with the decode, so a few slots suffice and
# bound the pinned host memory.
_TRANSFER_SLOTS = 4
# Frame conversion threads for the direct planar route. The muxing thread
# shares the worker's GIL with the decode loop; one converter thread keeps
# up with the decode without competing with it further.
_FRAME_CONVERSION_WORKERS = 1


def stream_mp4_enabled() -> bool:
    """Whether a request that asks for a streamed response MP4 gets one."""
    raw = os.environ.get(MINIMAX_H3_STREAM_MP4_ENV, "").strip().lower()
    if raw in _ENABLED_VALUES:
        return True
    if raw in _DISABLED_VALUES:
        return False
    raise ValueError(f"{MINIMAX_H3_STREAM_MP4_ENV} must be 0 or 1, got {raw!r}")


# Reject an invalid switch value when the model is imported rather than on
# every request; requests still read it, so it can change between them.
stream_mp4_enabled()


def resolve_stream_mp4_request(extra_args: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return the API's response-encoding description if this request may stream it."""
    spec = (extra_args or {}).get(STREAMING_RESPONSE_ENCODING_KEY)
    if not isinstance(spec, Mapping) or not stream_mp4_enabled():
        return None
    if spec.get("frame_path") not in STREAMING_RESPONSE_FRAME_PATHS:
        return None
    options = spec.get("video_codec_options")
    if options is not None and not isinstance(options, Mapping):
        return None
    # libx264's automatic thread (and slice) count follows the CPUs the
    # encoding thread may use; a different count would change the bitstream.
    cpus = usable_cpu_count()
    if spec.get("affinity_cpus") != cpus:
        logger.info_once(
            "MiniMax-H3 response MP4 is encoded by the API: its encoder sees %s CPUs, this worker %d",
            spec.get("affinity_cpus"),
            cpus,
        )
        return None
    return dict(spec)


class MiniMaxH3ResponseMP4Stream:
    """Feed the streaming decode's finished frames, in order, to one response MP4 encoder.

    Install an instance as the decode's frame sink (``publish_decoded_frames``).
    Each call copies the new frames, cropped to the requested size and laid out
    as the ``(T, H, W, 3)`` frames the API would receive, into a pinned host
    slot on a side stream; the muxing thread waits for the copy, so the decode
    never does. A failure only marks the stream failed: the sink runs inside
    the distributed decode, where raising would leave the peers hanging in the
    next collective. :meth:`close` or :meth:`result` then reports the failure
    and the caller returns the decoded frames as before.
    """

    def __init__(
        self,
        spec: Mapping[str, Any],
        *,
        height: int,
        width: int,
        fps: int | float,
        audio_channels: int,
        audio_sample_rate: int,
        device: torch.device,
    ) -> None:
        self._height = int(height)
        self._width = int(width)
        self._ring = PinnedChunkRing(depth=_TRANSFER_SLOTS, device=device)
        self._leases: list[ChunkLease] = []
        self._dec: torch.Tensor | None = None
        self._next_frame = 0
        self._error: BaseException | None = None
        self._encoder = StreamingVideoBytesEncoder(
            width=self._width,
            height=self._height,
            fps=fps,
            frame_path=spec["frame_path"],
            audio_channels=audio_channels,
            audio_sample_rate=audio_sample_rate,
            video_codec_options=spec.get("video_codec_options"),
            frame_conversion_workers=_FRAME_CONVERSION_WORKERS,
            max_pending=_TRANSFER_SLOTS,
        )

    def __call__(self, dec: torch.Tensor, start: int, end: int) -> None:
        """Publish ``dec[:, :, start:end]``, which the decode has just finalized."""
        if self._error is not None:
            return
        try:
            if self._dec is None:
                self._dec = dec
            elif dec is not self._dec:
                raise RuntimeError("the decode switched output buffers mid-stream")
            if start != self._next_frame or end <= start:
                raise RuntimeError(f"frames {start}:{end} are out of order after frame {self._next_frame}")
            if dec.ndim != 5 or dec.shape[0] != 1 or dec.shape[1] != 3 or dec.dtype != torch.uint8:
                raise RuntimeError(f"unexpected decode buffer {tuple(dec.shape)} {dec.dtype}")
            self._next_frame = end
            # The layout _prepare_minimax_h3_video_output hands the API, per chunk.
            frames = dec[0, :, start:end, : self._height, : self._width].permute(1, 2, 3, 0).contiguous()
            lease = self._ring.transfer(frames, readers=1)
            self._leases.append(lease)
            self._encoder.push(lease.wait, on_consumed=lease.release)
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)

    def finish(self, video: torch.Tensor, audio: torch.Tensor) -> tuple[bytes, int] | None:
        """:meth:`close` and :meth:`result` in one call."""
        return self.result() if self.close(video, audio) else None

    def close(self, video: torch.Tensor, audio: torch.Tensor) -> bool:
        """Check the decode against the stream and hand the waveform to the encoder.

        ``video`` and ``audio`` are what the raw-frame path would return. The
        video must be exactly the frames that were streamed; the waveform is
        converted the way the result post-processing converts it for the API.
        Returns ``False`` when the caller must return the frames instead; keep
        them until :meth:`result` succeeds as well. The encoder muxes the audio
        track while the caller goes on.
        """
        try:
            if self._error is not None:
                raise self._error
            if self._dec is None:
                raise RuntimeError("the decode published no frames")
            expected = (1, 3, self._next_frame, self._height, self._width)
            if tuple(video.shape) != expected or self._dec.shape[2] != self._next_frame:
                raise RuntimeError(f"decoded video {tuple(video.shape)} was not fully streamed as {expected}")
            if not torch.equal(video, self._dec[..., : self._height, : self._width]):
                raise RuntimeError("decoded video differs from the streamed frames")
            self._encoder.close(audio.detach().float().cpu().numpy())
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)
            self.abort()
            return False
        self._dec = None
        return True

    def result(self) -> tuple[bytes, int] | None:
        """Wait for the MP4 after :meth:`close`: its bytes and frame count, or ``None`` to return the frames."""
        try:
            data = self._encoder.result()
            if self._encoder.frame_count != self._next_frame:
                raise RuntimeError(f"encoded {self._encoder.frame_count} of {self._next_frame} frames")
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)
            self.abort()
            return None
        self._leases.clear()
        return data, self._next_frame

    def abort(self) -> None:
        """Stop the encoder and wait for every copy still in flight."""
        self._ring.abort()
        self._encoder.abort()
        for lease in self._leases:
            lease.discard()
        self._leases.clear()
        self._dec = None

    def _fail(self, exc: BaseException) -> None:
        if self._error is None:
            self._error = exc
            logger.warning("MiniMax-H3 response MP4 streaming failed; returning raw frames: %s", exc)

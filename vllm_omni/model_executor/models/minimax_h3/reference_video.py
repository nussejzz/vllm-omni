# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax H3 Ref2VA reference-video preparation."""

from __future__ import annotations

import fcntl
import io
import json
import math
import os
import re
import subprocess
import threading
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO, Any

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.errors import OmniClientError

logger = init_logger(__name__)

MINIMAX_H3_PREPARED_REFERENCE_VIDEOS_KEY = "minimax_h3_prepared_reference_videos"
# Transitional switch for direct reference-video frame decoding: unset uses
# the default worker count, 0 restores the lossless prepared.mp4 round trip
# (and the other reference prefetches), N decodes each request's reference
# videos with N ffmpeg processes split across them, at least one per video.
MINIMAX_H3_REFERENCE_DECODE_WORKERS_ENV = "VLLM_OMNI_H3_REFERENCE_DECODE_WORKERS"
_MAX_DEFAULT_REFERENCE_DECODE_WORKERS = 8
_REFERENCE_FRAME_PIPE_BYTES = 1 << 20
# Split decodes log at warning level with level tags: ffmpeg reports a
# mid-stream frame-property change only as a warning, and only the error
# lines are forwarded, as the single-process command would print them.
_FFMPEG_LOG_LEVEL_TAG = re.compile(rb"\[(trace|debug|verbose|info|warning|error|fatal|panic)\] ")
_FFMPEG_ERROR_LEVELS = frozenset((b"error", b"fatal", b"panic"))
_FFMPEG_FRAME_PROPERTY_CHANGE = b"Changing video frame properties on the fly"
_FFMPEG_ERROR_LINES_KEPT = 32


def serialize_prepared_reference_videos(items: list[dict[str, Any]], artifact_dir: str) -> str:
    fields = (
        "original_path",
        "prepared_path",
        "input_has_audio",
        "width",
        "height",
        "start_time_seconds",
        "duration_seconds",
        "audio_duration_seconds",
    )
    return json.dumps(
        {
            "artifact_dir": artifact_dir,
            "videos": [{field: item[field] for field in fields} for item in items],
        },
        separators=(",", ":"),
    )


def deserialize_prepared_reference_videos(value: str) -> tuple[str, list[dict[str, Any]]]:
    try:
        payload = json.loads(value)
        artifact_dir = payload["artifact_dir"]
        videos = payload["videos"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid MiniMax H3 prepared-reference-video descriptor") from exc
    if not isinstance(artifact_dir, str) or not isinstance(videos, list):
        raise ValueError("invalid MiniMax H3 prepared-reference-video descriptor")
    required = {
        "original_path",
        "prepared_path",
        "input_has_audio",
        "width",
        "height",
        "start_time_seconds",
        "duration_seconds",
        "audio_duration_seconds",
    }
    if any(not isinstance(item, dict) or set(item) != required for item in videos):
        raise ValueError("invalid MiniMax H3 prepared-reference-video descriptor")
    return artifact_dir, videos


MINIMAX_H3_FPS = 24.0
MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS = 2.0
MINIMAX_H3_QWEN_TEMPORAL_PATCH = 2
MINIMAX_H3_CANVAS_MULTIPLE = 32
MINIMAX_H3_BASE_SHORT_EDGE = 768
MINIMAX_H3_MAX_PIXELS = 768 * 1344
MINIMAX_H3_MIN_REFERENCE_DIMENSION = 256
MINIMAX_H3_MAX_REFERENCE_DIMENSION = 5760
MINIMAX_H3_MIN_REFERENCE_FPS = 23.976
MINIMAX_H3_MAX_REFERENCE_FPS = 60.0
MINIMAX_H3_MIN_REFERENCE_DURATION = 2.0
MINIMAX_H3_MAX_REFERENCE_DURATION = 15.0
MINIMAX_H3_MAX_REFERENCE_TOTAL_DURATION = 15.0
# ffprobe reports container durations at a finer precision than the official
# example's rounded values. Trim only this small probe-rounding excess.
MINIMAX_H3_REFERENCE_DURATION_EPSILON = 1e-2
MINIMAX_H3_MAX_VIDEO_BYTES = 50 * 1024 * 1024
MINIMAX_H3_MAX_AUDIO_BYTES = 15 * 1024 * 1024
MINIMAX_H3_VIDEO_FORMATS = frozenset({"mp4", "mov"})
MINIMAX_H3_VIDEO_CODECS = frozenset({"h264", "h265", "hevc"})
MINIMAX_H3_AUDIO_FORMATS = frozenset({"wav", "mp3"})
MINIMAX_H3_AUDIO_CODECS = frozenset({"aac", "mp3"})


def _nearest_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def _probe_video(path: str) -> dict[str, Any]:
    show_entries = (
        "stream=codec_type,codec_name,width,height,r_frame_rate,nb_read_frames,nb_frames,duration,"
        "sample_aspect_ratio:format=format_name,size,duration"
    )
    command = ["ffprobe", "-v", "error", "-show_entries", show_entries, "-of", "json", path]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    probe = json.loads(result.stdout)
    streams = probe.get("streams") or []
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    if not video_streams:
        raise OmniClientError(f"media has no video stream: {path}")
    stream = video_streams[0]
    raw_count = stream.get("nb_read_frames") or stream.get("nb_frames")
    if raw_count in (None, "", "N/A"):
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", *command[3:]],
            check=True,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        probe = json.loads(result.stdout)
        streams = probe.get("streams") or []
        video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
        if not video_streams:
            raise OmniClientError(f"media has no video stream: {path}")
        stream = video_streams[0]
        raw_count = stream.get("nb_read_frames") or stream.get("nb_frames")
    try:
        numerator, denominator = str(stream["r_frame_rate"]).split("/", 1)
        fps = float(numerator) / float(denominator)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise OmniClientError(f"cannot determine video FPS: {path}") from exc
    if not math.isfinite(fps) or fps <= 0:
        raise OmniClientError(f"cannot determine video FPS: {path}")
    if raw_count in (None, "", "N/A"):
        raise OmniClientError(f"cannot determine video frame count: {path}")
    raw_duration = stream.get("duration")
    if raw_duration in (None, "", "N/A"):
        raw_duration = probe.get("format", {}).get("duration")
    duration = float(raw_duration) if raw_duration not in (None, "", "N/A") else int(raw_count) / fps
    format_info = probe.get("format") or {}
    format_names = tuple(
        name.strip().lower() for name in str(format_info.get("format_name", "")).split(",") if name.strip()
    )
    raw_size = format_info.get("size")
    file_size = int(raw_size) if raw_size not in (None, "", "N/A") else os.path.getsize(path)
    audio_codecs = tuple(
        str(stream.get("codec_name", "")).lower()
        for stream in streams
        if stream.get("codec_type") == "audio" and stream.get("codec_name")
    )
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "frame_count": int(raw_count),
        "duration": duration,
        "format_names": format_names,
        "video_codec": str(stream.get("codec_name", "")).lower(),
        "audio_codecs": audio_codecs,
        "file_size": file_size,
        # ffmpeg's constant-rate output treats single-stream inputs differently.
        "stream_count": len(streams),
    }


def _validate_reference_video_metadata(
    meta: dict[str, Any],
    *,
    index: int,
    source: str,
    max_duration: float = MINIMAX_H3_MAX_REFERENCE_DURATION,
) -> None:
    width = int(meta["width"])
    height = int(meta["height"])
    if (
        min(width, height) < MINIMAX_H3_MIN_REFERENCE_DIMENSION
        or max(width, height) > MINIMAX_H3_MAX_REFERENCE_DIMENSION
    ):
        raise OmniClientError(f"reference video {index} dimensions must be in [256, 5760] pixels, got {width}x{height}")
    ratio = width / height
    if not 0.4 <= ratio <= 2.5:
        raise OmniClientError(f"reference video {index} aspect ratio must be in [0.4, 2.5], got {width}x{height}")

    fps = float(meta["fps"])
    if not MINIMAX_H3_MIN_REFERENCE_FPS <= fps <= MINIMAX_H3_MAX_REFERENCE_FPS:
        raise OmniClientError(f"reference video {index} FPS must be in [23.976, 60], got {fps:.3f}")
    duration = float(meta["duration"])
    if not math.isfinite(duration) or not MINIMAX_H3_MIN_REFERENCE_DURATION <= duration <= max_duration:
        raise OmniClientError(
            f"reference video {index} duration must be in [2, {max_duration:g}] seconds, got {duration:.3f}"
        )

    raw_file_size = meta.get("file_size")
    file_size = int(raw_file_size) if raw_file_size is not None else os.path.getsize(source)
    if file_size > MINIMAX_H3_MAX_VIDEO_BYTES:
        raise OmniClientError(f"reference video {index} exceeds the 50 MiB size limit")
    format_names = set(meta.get("format_names", ()))
    if not format_names.intersection(MINIMAX_H3_VIDEO_FORMATS):
        formats = ", ".join(sorted(format_names)) or "unknown"
        raise OmniClientError(f"reference video {index} must use an MP4 or MOV container, got {formats}")
    video_codec = str(meta.get("video_codec", "")).lower()
    if video_codec not in MINIMAX_H3_VIDEO_CODECS:
        raise OmniClientError(f"reference video {index} must use H.264 or H.265 video, got {video_codec or 'unknown'}")
    audio_codecs = tuple(str(codec).lower() for codec in meta.get("audio_codecs", ()))
    invalid_audio = [codec for codec in audio_codecs if codec not in MINIMAX_H3_AUDIO_CODECS]
    if invalid_audio:
        raise OmniClientError(
            f"reference video {index} must use AAC or MP3 audio when audio is present, got {invalid_audio[0]}"
        )


def _audio_items(values: Any) -> list[Any]:
    if isinstance(values, (str, os.PathLike)):
        return [values]
    if isinstance(values, tuple) and len(values) == 2 and isinstance(values[1], (int, np.integer)):
        return [values]
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


def _probe_audio(path: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,duration:format=format_name,size,duration",
            "-of",
            "json",
            path,
        ],
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    probe = json.loads(result.stdout)
    streams = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == "audio"]
    if not streams:
        raise OmniClientError(f"media has no audio stream: {path}")
    stream = streams[0]
    raw_duration = stream.get("duration")
    if raw_duration in (None, "", "N/A"):
        raw_duration = (probe.get("format") or {}).get("duration")
    if raw_duration in (None, "", "N/A"):
        raise OmniClientError(f"cannot determine audio duration: {path}")
    format_info = probe.get("format") or {}
    format_names = tuple(
        name.strip().lower() for name in str(format_info.get("format_name", "")).split(",") if name.strip()
    )
    raw_size = format_info.get("size")
    file_size = int(raw_size) if raw_size not in (None, "", "N/A") else os.path.getsize(path)
    return {
        "duration": float(raw_duration),
        "format_names": format_names,
        "codec": str(stream.get("codec_name", "")).lower(),
        "file_size": file_size,
    }


def validate_reference_audio_files(
    values: Any,
    *,
    probe: Callable[[str], dict[str, Any]] | None = None,
) -> None:
    """Validate path-backed standalone audio references against the H3 spec.

    ``probe`` replaces :func:`_probe_audio`, e.g. to consume a prefetched probe.
    """
    total_duration = 0.0
    for index, value in enumerate(_audio_items(values)):
        if not isinstance(value, (str, os.PathLike)):
            continue
        source = str(value)
        meta = (probe or _probe_audio)(source)
        total_duration += _validate_reference_audio_metadata(meta, index=index)
    if total_duration > MINIMAX_H3_MAX_REFERENCE_TOTAL_DURATION:
        raise OmniClientError("reference audio must be at most 15 seconds in total")


def _validate_reference_audio_metadata(meta: dict[str, Any], *, index: int) -> float:
    """Check one probed standalone audio file against the H3 spec; return its duration."""
    file_size = int(meta["file_size"])
    if file_size > MINIMAX_H3_MAX_AUDIO_BYTES:
        raise OmniClientError(f"reference audio {index} exceeds the 15 MiB size limit")
    if not set(meta["format_names"]).intersection(MINIMAX_H3_AUDIO_FORMATS):
        formats = ", ".join(meta["format_names"]) or "unknown"
        raise OmniClientError(f"reference audio {index} must be WAV or MP3, got {formats}")
    codec = str(meta["codec"]).lower()
    if codec not in MINIMAX_H3_AUDIO_CODECS and not codec.startswith("pcm_"):
        raise OmniClientError(f"reference audio {index} must use MP3 or PCM audio, got {codec or 'unknown'}")
    duration = float(meta["duration"])
    if (
        not math.isfinite(duration)
        or not MINIMAX_H3_MIN_REFERENCE_DURATION <= duration <= MINIMAX_H3_MAX_REFERENCE_DURATION
    ):
        raise OmniClientError(f"reference audio {index} duration must be in [2, 15] seconds, got {duration:.3f}")
    return duration


def validate_reference_audio_waveforms(
    values: list[tuple[torch.Tensor, int]],
    *,
    max_seconds: float = MINIMAX_H3_MAX_REFERENCE_TOTAL_DURATION,
) -> None:
    """Validate in-memory standalone audio references using sample counts."""
    total_duration = 0.0
    for index, (waveform, sample_rate) in enumerate(values):
        if int(sample_rate) <= 0:
            raise OmniClientError(f"reference audio {index} has an invalid sample rate")
        samples = int(waveform.shape[-1])
        duration = samples / int(sample_rate)
        if not MINIMAX_H3_MIN_REFERENCE_DURATION <= duration <= max_seconds:
            raise OmniClientError(
                f"reference audio {index} duration must be in [2, {max_seconds:g}] seconds, got {duration:.3f}"
            )
        total_duration += duration
    if total_duration > max_seconds:
        raise OmniClientError(f"reference audio must be at most {max_seconds:g} seconds in total")


def _reference_video_shape(width: int, height: int) -> tuple[int, int]:
    """Fit large references to the video canvas without enlarging small inputs."""
    if (
        min(width, height) < MINIMAX_H3_MIN_REFERENCE_DIMENSION
        or max(width, height) > MINIMAX_H3_MAX_REFERENCE_DIMENSION
    ):
        raise OmniClientError(f"reference video dimensions must be in [256, 5760] pixels, got {width}x{height}")
    ratio = float(width) / float(height)
    if not 0.4 <= ratio <= 2.5:
        raise OmniClientError(f"reference video aspect ratio must be in [0.4, 2.5], got {width}x{height}")
    if ratio >= 1.0:
        target_width = MINIMAX_H3_BASE_SHORT_EDGE * ratio
        target_height = float(MINIMAX_H3_BASE_SHORT_EDGE)
    else:
        target_width = float(MINIMAX_H3_BASE_SHORT_EDGE)
        target_height = MINIMAX_H3_BASE_SHORT_EDGE / ratio
    area = target_width * target_height
    if area > MINIMAX_H3_MAX_PIXELS:
        scale = math.sqrt(MINIMAX_H3_MAX_PIXELS / area)
        target_width *= scale
        target_height *= scale
    canvas_width = _nearest_multiple(target_width, MINIMAX_H3_CANVAS_MULTIPLE)
    canvas_height = _nearest_multiple(target_height, MINIMAX_H3_CANVAS_MULTIPLE)
    if width * height < canvas_width * canvas_height:
        return (
            _nearest_multiple(width, MINIMAX_H3_CANVAS_MULTIPLE),
            _nearest_multiple(height, MINIMAX_H3_CANVAS_MULTIPLE),
        )
    return canvas_width, canvas_height


def _reference_video_filters(
    width: int,
    height: int,
    *,
    cover: bool = False,
    pad_last_frame: bool = False,
    frame_filters: Sequence[str] = (),
) -> list[str]:
    """Return the ffmpeg filter chain that maps a source onto the H3 frame grid.

    ``frame_filters`` run on the 24 fps frames, before they are resized.
    """
    if cover:
        # Fill the canvas and centre-crop the overflow instead of stretching,
        # as Open-H3-Regenerate prepares its drafts.
        resize = [
            f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=bilinear",
            f"crop={width}:{height}",
        ]
    else:
        resize = [f"scale={width}:{height}:flags=lanczos"]
    filters = [f"fps={MINIMAX_H3_FPS:g}", *frame_filters, *resize, "setsar=1"]
    if pad_last_frame:
        # Keep cloning the final converted frame until -frames:v reaches the
        # requested target; longer inputs are deterministically trimmed by the
        # same -frames:v limit without enabling this padding filter.
        filters.append("tpad=stop_mode=clone:stop=-1")
    return filters


def _transcode_reference_video(
    source: str,
    *,
    target_width: int,
    target_height: int,
    target_frame_count: int,
    workdir: str,
    start_time_seconds: float = 0.0,
    duration_seconds: float | None = None,
    pad_last_frame: bool = False,
    cover: bool = False,
) -> str:
    output = str(Path(workdir) / "prepared.mp4")
    duration_args = ["-t", f"{float(duration_seconds):.6f}"] if duration_seconds is not None else []
    frame_count_args = ["-frames:v", str(int(target_frame_count))] if target_frame_count > 0 else []
    filters = _reference_video_filters(
        target_width,
        target_height,
        cover=cover,
        pad_last_frame=pad_last_frame,
    )
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{float(start_time_seconds):.6f}",
        "-i",
        source,
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        ",".join(filters),
        *duration_args,
        *frame_count_args,
        "-metadata:s:v:0",
        "rotate=0",
        # Keep the transformed RGB pixels lossless. The same prepared
        # stream is decoded by Qwen3VL and the video VAE; a yuv420p
        # intermediate would make the two consumers see different
        # pixels from the model-card reference recipe.
        "-c:v",
        "libx264rgb",
        "-crf",
        "0",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "rgb24",
        output,
    ]
    subprocess.run(command, check=True, stdin=subprocess.DEVNULL)
    return output


def reference_frame_workers() -> int:
    """Return the per-request ffmpeg process target for direct reference-video decoding.

    ``0`` keeps the lossless ``prepared.mp4`` round trip. When unset, one
    process per eight usable CPUs is used, at most eight. A request with
    several reference videos splits the target, at least one per video.
    """
    raw = os.environ.get(MINIMAX_H3_REFERENCE_DECODE_WORKERS_ENV, "").strip()
    if not raw:
        try:
            cpus = len(os.sched_getaffinity(0))
        except AttributeError:  # sched_getaffinity is Linux-only.
            cpus = os.cpu_count() or 1
        return min(_MAX_DEFAULT_REFERENCE_DECODE_WORKERS, max(1, cpus // 8))
    try:
        workers = int(raw)
    except ValueError:
        workers = -1
    if workers < 0:
        raise ValueError(f"{MINIMAX_H3_REFERENCE_DECODE_WORKERS_ENV} must be a non-negative integer, got {raw!r}")
    return workers


# Reject an invalid switch value when the model is imported rather than on
# every request; requests still read it, so it can change between them.
reference_frame_workers()


def _enlarge_pipe(pipe: IO[bytes] | None) -> None:
    # Best effort: a larger pipe lets ffmpeg run ahead of the reader thread.
    set_pipe_size = getattr(fcntl, "F_SETPIPE_SZ", None)  # Linux-only.
    if pipe is None or set_pipe_size is None:
        return
    try:
        fcntl.fcntl(pipe.fileno(), set_pipe_size, _REFERENCE_FRAME_PIPE_BYTES)
    except OSError:
        pass


def _write_stderr(data: bytes) -> None:
    # Where an inherited ffmpeg stderr would have written.
    try:
        os.write(2, data)
    except OSError:
        pass


class ReferenceVideoFrames:
    """One reference video decoded straight to RGB frames by background ffmpeg processes.

    The frames are the ones :func:`_transcode_reference_video` stores in its
    lossless ``prepared.mp4``: the same seek, filter chain and limits, piped
    as rgb24 instead of being encoded and decoded again. One process with
    ffmpeg's default output sync reproduces that transcode exactly.

    With ``K`` workers, output frame ``j`` comes from process ``j % K``, which
    selects it on the 24 fps grid and passes it through with ``-vsync 0``.
    The steps ffmpeg would apply to the unsplit stream are made explicit:

    * the output ``-t`` trims from the first frame that reaches it, which
      differs per process, so a ``trim`` runs before the split;
    * constant-rate output repeats a first frame that lands after time 0
      down to time 0, except for single-stream inputs, which start at that
      frame. For multi-stream sources a second ``fps`` with ``start_time=0``
      makes those repeats, and ``-t``'s output-time limit becomes a frame
      count, because ffmpeg counts the repeats against it.

    ``stream_count`` (all streams in the source container) selects between
    the two; without it the decode runs in one process.

    A source whose frame size or pixel format changes mid-stream makes
    ffmpeg rebuild the filter graph, which restarts every split process's
    frame numbering. Split processes therefore keep their graph
    (``-reinit_filter 0``) and watch their log for ffmpeg's frame-property
    warning; on it they stop, and :meth:`result` decodes the video again in
    one process, whose graph is rebuilt exactly as the transcode's is.

    One reader thread per process copies every frame into its slot of a
    preallocated tensor. Callers must call :meth:`result` or :meth:`cancel`.
    """

    def __init__(
        self,
        source: str,
        *,
        width: int,
        height: int,
        frame_count: int,
        start_time_seconds: float = 0.0,
        duration_seconds: float | None = None,
        workers: int = 1,
        stream_count: int | None = None,
    ) -> None:
        if int(frame_count) <= 0:
            raise ValueError("reference video frame decoding requires a positive target frame count")
        self.source = source
        self.frame_count = int(frame_count)
        self._options: dict[str, Any] = {
            "width": int(width),
            "height": int(height),
            "frame_count": self.frame_count,
            "start_time_seconds": start_time_seconds,
            "duration_seconds": duration_seconds,
            "stream_count": stream_count,
        }
        self._repeat_late_start = stream_count is not None and int(stream_count) != 1
        self._frame_limit = self.frame_count
        if self._repeat_late_start and duration_seconds is not None:
            # ffmpeg keeps output frame j while j / 24 < -t; it parses -t as
            # whole microseconds.
            micros = round(float(f"{float(duration_seconds):.6f}") * 1_000_000)
            fps = int(MINIMAX_H3_FPS)
            self._frame_limit = min(self.frame_count, -(-micros * fps // 1_000_000))
        self.workers = 1 if stream_count is None else max(1, min(int(workers), self._frame_limit))
        # True once a split decode met a frame-property change and ran again in one process.
        self.serial_fallback = False
        self._frames: torch.Tensor | None = torch.empty(
            (self.frame_count, int(height), int(width), 3),
            dtype=torch.uint8,
        )
        self._counts = [0] * self.workers
        self._errors: list[BaseException | None] = [None] * self.workers
        self._error_logs: list[deque[bytes]] = [deque(maxlen=_FFMPEG_ERROR_LINES_KEPT) for _ in range(self.workers)]
        self._property_changed = False
        self._processes: list[subprocess.Popen[bytes]] = []
        self._threads: list[threading.Thread] = []
        self._consumed = False
        split = self.workers > 1
        frames = self._frames.numpy()
        try:
            for worker in range(self.workers):
                process = subprocess.Popen(
                    self._command(
                        worker,
                        width=int(width),
                        height=int(height),
                        start_time_seconds=start_time_seconds,
                        duration_seconds=duration_seconds,
                    ),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    # One process logs straight to stderr, like the transcode.
                    stderr=subprocess.PIPE if split else None,
                    bufsize=0,
                )
                self._processes.append(process)
                _enlarge_pipe(process.stdout)
                self._start_thread(self._read, worker, process.stdout, frames)
                if split:
                    self._start_thread(self._watch_log, worker, process)
        except BaseException:
            self._stop()
            raise
        logger.debug("Decoding MiniMax H3 reference video %s with %d ffmpeg process(es)", source, self.workers)

    def _start_thread(self, target: Callable[..., None], worker: int, *args: Any) -> None:
        thread = threading.Thread(
            target=target,
            args=(worker, *args),
            name=f"minimax_h3_reference_frames_{worker}",
            daemon=True,
        )
        thread.start()
        self._threads.append(thread)

    def _command(
        self,
        worker: int,
        *,
        width: int,
        height: int,
        start_time_seconds: float,
        duration_seconds: float | None,
    ) -> list[str]:
        frame_filters: list[str] = []
        split = self.workers > 1
        if split:
            if duration_seconds is not None:
                frame_filters.append(f"trim=duration={float(duration_seconds):.6f}")
            if self._repeat_late_start:
                frame_filters.append(f"fps={MINIMAX_H3_FPS:g}:start_time=0")
            frame_filters.append(f"select=eq(mod(n\\,{self.workers})\\,{worker})")
        command = ["ffmpeg", "-nostdin", "-loglevel", "level+warning" if split else "error"]
        if split:
            # Keep the graph on a frame-property change and report it (see
            # the class docstring). None of these filters is slice-threaded,
            # so skip ffmpeg's per-CPU filter thread pool.
            command.extend(["-filter_threads", "1", "-reinit_filter", "0"])
        command += [
            "-ss",
            f"{float(start_time_seconds):.6f}",
            "-i",
            self.source,
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            ",".join(_reference_video_filters(width, height, frame_filters=frame_filters)),
        ]
        if duration_seconds is not None and not (split and self._repeat_late_start):
            command.extend(["-t", f"{float(duration_seconds):.6f}"])
        frame_limit = self._frame_limit if split else self.frame_count
        command.extend(["-frames:v", str(len(range(worker, frame_limit, self.workers)))])
        if split:
            # Keep the selected frames' 24 fps timestamps; constant-rate output
            # would duplicate frames into the gaps between them.
            command.extend(["-vsync", "0"])
        command.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        return command

    def _read(self, worker: int, pipe: io.RawIOBase, frames: np.ndarray) -> None:
        count = 0
        try:
            with pipe:
                for index in range(worker, self.frame_count, self.workers):
                    view = memoryview(frames[index]).cast("B")
                    filled = 0
                    while filled < len(view):
                        size = pipe.readinto(view[filled:])
                        if not size:
                            break
                        filled += size
                    if filled == 0:
                        break
                    if filled != len(view):
                        raise OmniClientError(
                            f"ffmpeg returned a partial reference video frame ({filled}/{len(view)} bytes): "
                            f"{self.source}"
                        )
                    count += 1
                else:
                    if pipe.read(1):
                        raise OmniClientError(
                            f"ffmpeg returned more reference video frames than requested: {self.source}"
                        )
        except BaseException as exc:  # Raised by result().
            self._errors[worker] = exc
        finally:
            self._counts[worker] = count

    def _watch_log(self, worker: int, process: subprocess.Popen[bytes]) -> None:
        """Read a split process's log: keep and forward its errors, stop it on a frame-property change."""
        pipe = process.stderr
        assert pipe is not None
        forward = False
        pending = b""
        with pipe:
            while chunk := pipe.read(1 << 16):
                *lines, pending = (pending + chunk).split(b"\n")
                for line in lines:
                    forward = self._log_line(worker, process, line + b"\n", forward)
            if pending:
                self._log_line(worker, process, pending, forward)

    def _log_line(self, worker: int, process: subprocess.Popen[bytes], line: bytes, forward: bool) -> bool:
        tag = _FFMPEG_LOG_LEVEL_TAG.search(line)
        if tag is not None:
            # Untagged lines continue the previous message.
            forward = tag.group(1) in _FFMPEG_ERROR_LEVELS
            line = line[: tag.start()] + line[tag.end() :]
        if _FFMPEG_FRAME_PROPERTY_CHANGE in line:
            # One process would rebuild its graph here; result() decodes again.
            self._property_changed = True
            process.kill()
        elif forward:
            self._error_logs[worker].append(line)
            if worker == 0:
                # Every process decodes the same stream; print its errors once.
                _write_stderr(line)
        return forward

    def _stop(self) -> None:
        for process in self._processes:
            if process.poll() is None:
                process.kill()
        for process in self._processes:
            process.wait()
        for thread in self._threads:
            thread.join()
        for process in self._processes:
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
        self._frames = None

    def result(self) -> torch.Tensor:
        """Wait for the decode and return its ``uint8 [T, H, W, 3]`` frames.

        A source that ends early yields fewer than ``frame_count`` frames, as
        its prepared video would; nothing is padded.
        """
        if self._consumed:
            raise RuntimeError("reference video frames were already consumed")
        self._consumed = True
        try:
            for thread in self._threads:
                thread.join()
            returncodes = [process.wait() for process in self._processes]
        except BaseException:
            self._stop()
            raise
        frames, self._frames = self._frames, None
        assert frames is not None
        if self._property_changed:
            del frames
            logger.warning(
                "MiniMax H3 reference video %s changes frame size or pixel format mid-stream; "
                "decoding it again in one ffmpeg process",
                self.source,
            )
            self.serial_fallback = True
            return ReferenceVideoFrames(self.source, **self._options, workers=1).result()
        for worker, (returncode, process) in enumerate(zip(returncodes, self._processes)):
            if returncode:
                failure = subprocess.CalledProcessError(
                    returncode,
                    process.args,
                    stderr=b"".join(self._error_logs[worker]) or None,
                )
                # A reader that saw ffmpeg break the frame protocol closes the
                # pipe, which also fails ffmpeg; keep the reader's reason.
                cause = self._errors[worker] or next((error for error in self._errors if error is not None), None)
                raise failure from cause
        for error in self._errors:
            if error is not None:
                raise error
        total = sum(self._counts)
        if total <= 0:
            raise OmniClientError(f"video has no frames: {self.source}")
        # Frame n comes from process n % K, so a complete prefix of the frames
        # gives process k exactly ceil((total - k) / K) of them.
        if any(count != -(-(total - worker) // self.workers) for worker, count in enumerate(self._counts)):
            raise OmniClientError(f"reference video decode returned inconsistent frame counts {self._counts}")
        return frames if total == self.frame_count else frames[:total].clone()

    def cancel(self) -> None:
        """Stop the decode and discard its frames; a no-op once consumed."""
        if not self._consumed:
            self._consumed = True
            self._stop()


def start_reference_video_frames(
    source: str,
    *,
    target_width: int,
    target_height: int,
    target_frame_count: int,
    start_time_seconds: float = 0.0,
    duration_seconds: float | None = None,
    workers: int = 1,
    stream_count: int | None = None,
) -> ReferenceVideoFrames:
    """Start decoding the frames :func:`_transcode_reference_video` would prepare.

    ``stream_count`` is the source's ``_probe_video`` ``"stream_count"``; it
    is required to split the decode across ``workers`` processes.
    """
    return ReferenceVideoFrames(
        source,
        width=target_width,
        height=target_height,
        frame_count=target_frame_count,
        start_time_seconds=start_time_seconds,
        duration_seconds=duration_seconds,
        workers=workers,
        stream_count=stream_count,
    )


def prepare_edit_video(
    value: Any,
    target_width: int,
    target_height: int,
    target_frame_count: int,
    workdir: str,
    cover: bool = False,
) -> dict[str, Any]:
    """Resize an edit source and clone its last frame to the target length.

    ``cover`` scales to fill the canvas and centre-crops instead of stretching.
    """
    if not isinstance(value, str | os.PathLike):
        raise OmniClientError("MiniMax H3 edit video input must be a single file path")
    source = str(value)
    source_meta = _probe_video(source)
    if int(source_meta.get("frame_count", 0)) <= 0:
        raise OmniClientError(f"video has no frames: {source}")

    Path(workdir).mkdir(parents=True, exist_ok=True)
    prepared_path = _transcode_reference_video(
        source,
        target_width=target_width,
        target_height=target_height,
        target_frame_count=target_frame_count,
        workdir=workdir,
        pad_last_frame=True,
        cover=cover,
    )

    return {
        "original_path": source,
        "prepared_path": prepared_path,
        "input_has_audio": bool(source_meta.get("audio_codecs")),
        "width": target_width,
        "height": target_height,
        "frame_count": target_frame_count,
    }


def prepare_reference_videos(
    values: Any,
    *,
    target_frame_count: int,
    workdir: str,
    start_time_seconds: Any = None,
    decode_frames: bool = False,
    frame_workers: int = 1,
    max_total_seconds: float = MINIMAX_H3_MAX_REFERENCE_TOTAL_DURATION,
) -> list[dict[str, Any]]:
    """Validate the Ref2VA reference videos, then prepare their frames.

    By default each video is transcoded to ``prepared_path``. ``decode_frames``
    instead starts a :class:`ReferenceVideoFrames` job per video under
    ``"frames"`` (``prepared_path`` is ``None``), each with
    ``max(1, frame_workers // len(videos))`` ffmpeg processes, all running
    at once; the caller must consume or cancel every job.
    """
    if isinstance(values, (str, os.PathLike)):
        values = [values]
    if not isinstance(values, (list, tuple)) or not values:
        raise OmniClientError("MiniMax H3 Ref2VA video input must be a path or a non-empty list of paths")
    if isinstance(start_time_seconds, (list, tuple)):
        start_times = list(start_time_seconds)
    else:
        start_times = [start_time_seconds] * len(values)
    if len(start_times) != len(values):
        raise OmniClientError("start_time_seconds must contain one value per reference video")

    # Validate every video before any ffmpeg work starts.
    segments: list[tuple[str, dict[str, Any], float, float, int, int]] = []
    total_duration = 0.0
    for index, value in enumerate(values):
        item_start = start_times[index]
        if isinstance(value, dict):
            item_start = value.get("start_time_seconds", item_start)
            value = value.get("path", value.get("video_path", value.get("video_url")))
        if not isinstance(value, (str, os.PathLike)):
            raise OmniClientError(
                f"MiniMax H3 Ref2VA video references require file paths, got item {index}: {type(value)!r}"
            )
        source = str(value)
        meta = _probe_video(source)
        _validate_reference_video_metadata(meta, index=index, source=source, max_duration=max_total_seconds)
        start = 0.0 if item_start is None else float(item_start)
        if not math.isfinite(start) or start < 0 or start >= float(meta["duration"]):
            raise OmniClientError(f"reference video {index} start_time_seconds is outside the source duration")
        duration = float(meta["duration"]) - start
        if duration < MINIMAX_H3_MIN_REFERENCE_DURATION:
            raise OmniClientError(
                f"reference video {index} must provide at least 2 seconds after start_time_seconds, got {duration:.3f}"
            )
        remaining_duration = max_total_seconds - total_duration
        if duration > remaining_duration:
            if duration - remaining_duration > MINIMAX_H3_REFERENCE_DURATION_EPSILON:
                raise OmniClientError(f"reference videos must be at most {max_total_seconds:g} seconds in total")
            duration = remaining_duration
            if duration < MINIMAX_H3_MIN_REFERENCE_DURATION:
                raise OmniClientError(f"reference videos must be at most {max_total_seconds:g} seconds in total")
        total_duration += duration
        width, height = _reference_video_shape(meta["width"], meta["height"])
        segments.append((source, meta, start, duration, width, height))

    decode_frames = decode_frames and target_frame_count > 0
    workers = max(1, int(frame_workers) // len(segments))
    prepared: list[dict[str, Any]] = []
    try:
        for index, (source, meta, start, duration, width, height) in enumerate(segments):
            frames: ReferenceVideoFrames | None = None
            prepared_path: str | None = None
            if decode_frames:
                frames = start_reference_video_frames(
                    source,
                    target_width=width,
                    target_height=height,
                    target_frame_count=target_frame_count,
                    start_time_seconds=start,
                    duration_seconds=duration,
                    workers=workers,
                    stream_count=meta.get("stream_count"),
                )
            else:
                item_workdir = Path(workdir) / f"video_{index}"
                item_workdir.mkdir(parents=True)
                prepared_path = _transcode_reference_video(
                    source,
                    target_width=width,
                    target_height=height,
                    target_frame_count=target_frame_count,
                    workdir=str(item_workdir),
                    start_time_seconds=start,
                    duration_seconds=duration,
                )
            item: dict[str, Any] = {
                "original_path": source,
                "prepared_path": prepared_path,
                "input_has_audio": bool(meta.get("audio_codecs")),
                "width": width,
                "height": height,
                "start_time_seconds": start,
                "duration_seconds": duration,
                # Reference video frames and their soundtrack are conditioned
                # only for the generated temporal extent. Keep the source
                # segment duration for metadata/validation, and expose the
                # bounded duration separately for audio extraction.
                "audio_duration_seconds": min(
                    duration,
                    float(target_frame_count) / MINIMAX_H3_FPS if target_frame_count > 0 else duration,
                ),
            }
            if frames is not None:
                item["frames"] = frames
            prepared.append(item)
    except BaseException:
        for item in prepared:
            if "frames" in item:
                item["frames"].cancel()
        raise
    return prepared


def load_video_frames(path: str) -> np.ndarray:
    try:
        import decord
    except ImportError:
        meta = _probe_video(path)
        return _decode_video_frames_ffmpeg(
            path,
            frame_count=int(meta["frame_count"]),
            width=int(meta["width"]),
            height=int(meta["height"]),
        )

    reader = decord.VideoReader(path)
    if len(reader) <= 0:
        raise OmniClientError(f"video has no frames: {path}")
    frames = reader.get_batch(list(range(len(reader))))
    return frames.asnumpy() if hasattr(frames, "asnumpy") else np.asarray(frames)


def _decode_video_frames_ffmpeg(
    path: str,
    *,
    frame_count: int,
    width: int,
    height: int,
    indices: list[int] | None = None,
) -> np.ndarray:
    frame_count = int(frame_count)
    width = int(width)
    height = int(height)
    if frame_count <= 0:
        raise OmniClientError(f"video has no frames: {path}")
    if width <= 0 or height <= 0:
        raise OmniClientError(f"video has invalid dimensions {width}x{height}: {path}")
    if indices is not None and (not indices or any(index < 0 or index >= frame_count for index in indices)):
        raise OmniClientError(f"invalid frame indices for {path}: {indices}")

    output_frame_count = frame_count if indices is None else len(indices)
    command = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-threads",
        "0",
        "-i",
        path,
        "-map",
        "0:v:0",
        "-an",
    ]
    if indices is not None:
        select = "+".join(f"eq(n\\,{index})" for index in indices)
        command.extend(["-vf", f"select={select}"])
    command.extend(
        [
            "-vsync",
            "0",
            "-frames:v",
            str(output_frame_count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
    frame_size = width * height * 3
    expected_size = output_frame_count * frame_size
    if len(result.stdout) != expected_size:
        raise OmniClientError(f"decoded {len(result.stdout)} bytes from {path}, expected {expected_size}")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        output_frame_count,
        height,
        width,
        3,
    )


def sample_reference_video_frames(
    prepared_path: str,
    *,
    decoded_frames: np.ndarray | None = None,
) -> dict[str, Any]:
    if decoded_frames is None:
        meta = _probe_video(prepared_path)
        frame_count = int(meta["frame_count"])
    else:
        decoded_frames = np.asarray(decoded_frames)
        if decoded_frames.ndim != 4 or decoded_frames.shape[-1] != 3 or len(decoded_frames) <= 0:
            raise OmniClientError(
                f"decoded reference video frames must have shape [T, H, W, 3], got {decoded_frames.shape}"
            )
        frame_count = len(decoded_frames)
    ratio = MINIMAX_H3_FPS / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS
    indices: list[int] = []
    cursor = 0.0
    while True:
        frame_index = int(round(cursor))
        if frame_index >= frame_count:
            break
        if not indices or frame_index > indices[-1]:
            indices.append(frame_index)
        cursor += ratio
    if not indices:
        raise OmniClientError(f"no frames sampled from {prepared_path}")

    if decoded_frames is None:
        decoded_frames = _decode_video_frames_ffmpeg(
            prepared_path,
            frame_count=frame_count,
            indices=indices,
            width=meta["width"],
            height=meta["height"],
        )
    else:
        decoded_frames = decoded_frames[indices]
    frames = [np.asarray(frame) for frame in decoded_frames]

    timestamps = [index / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS for index in range(len(indices))]
    timestamps += [timestamps[-1]] * ((-len(timestamps)) % MINIMAX_H3_QWEN_TEMPORAL_PATCH)
    block_timestamps = [
        (timestamps[index] + timestamps[index + MINIMAX_H3_QWEN_TEMPORAL_PATCH - 1]) / 2
        for index in range(
            0,
            len(timestamps),
            MINIMAX_H3_QWEN_TEMPORAL_PATCH,
        )
    ]
    return {
        "frames": frames,
        "block_timestamps": block_timestamps,
    }


def _soundfile_to_waveform(path: str) -> tuple[torch.Tensor, int]:
    import soundfile as sf

    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    # soundfile is (samples, channels); torchaudio.load is (channels, samples).
    waveform = torch.from_numpy(data).t().contiguous()
    return waveform, int(sample_rate)


def load_audio_file(path: str) -> tuple[torch.Tensor, int]:
    """Load an audio file as ``(waveform[C, T] float32, sample_rate)``.

    torchaudio 2.6+ routes ``load`` through TorchCodec, whose wheels do not load
    on aarch64 with a CPU-only torch (they link CUDA libraries that are absent).
    Fall back to soundfile; if the container is not libsndfile-readable (e.g.
    mp3/m4a/mp4), demux to wav with ffmpeg first so any ffmpeg-supported input
    works at its native sample rate.
    """
    try:
        import torchaudio

        return torchaudio.load(path)
    except (ImportError, RuntimeError, OSError):
        pass
    try:
        return _soundfile_to_waveform(path)
    except Exception:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="minimax_h3_audio_") as tmpdir:
            wav = str(Path(tmpdir) / "audio.wav")
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    path,
                    "-vn",
                    "-f",
                    "wav",
                    wav,
                ],
                check=True,
                stdin=subprocess.DEVNULL,
            )
            return _soundfile_to_waveform(wav)


_AUDIO_EXTRACTION_TIMEOUT_SECONDS = 120


def load_video_audio(
    path: str,
    *,
    start_time_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> tuple[torch.Tensor, int]:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="minimax_h3_video_audio_") as tmpdir:
        output = str(Path(tmpdir) / "audio.wav")
        command = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{float(start_time_seconds):.6f}",
            "-i",
            path,
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-f",
            "wav",
        ]
        if duration_seconds is not None:
            command.extend(["-t", f"{float(duration_seconds):.6f}"])
        command.append(output)
        subprocess.run(
            command,
            check=True,
            stdin=subprocess.DEVNULL,
            timeout=_AUDIO_EXTRACTION_TIMEOUT_SECONDS,
        )
        return load_audio_file(output)


__all__ = [
    "ReferenceVideoFrames",
    "load_audio_file",
    "load_video_audio",
    "load_video_frames",
    "prepare_edit_video",
    "prepare_reference_videos",
    "reference_frame_workers",
    "sample_reference_video_frames",
    "start_reference_video_frames",
    "validate_reference_audio_files",
    "validate_reference_audio_waveforms",
]

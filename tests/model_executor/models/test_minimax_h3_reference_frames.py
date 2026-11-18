# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Direct reference-video frame decoding must match the lossless prepared.mp4 round trip byte for byte."""

import hashlib
import os
import shutil
import subprocess

import numpy as np
import psutil
import pytest
import torch
from PIL import Image

from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.minimax_h3 import encoder_processing as processing
from vllm_omni.model_executor.models.minimax_h3 import reference_video

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TARGET_FRAMES = 124
WIDTH, HEIGHT = 160, 96
WORKER_COUNTS = (1, 3, 8)
# name: (lavfi size, fps, seconds, with audio, rotate metadata)
SOURCES = {
    "s30_audio": ("384x256", 30, 6.0, True, None),
    "s25_short": ("288x256", 25, 3.1, False, None),
    "s60_rotate": ("320x256", 60, 5.0, False, 90),
}


def _ffmpeg_encoders() -> str:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return ""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=False, timeout=30
    )
    return result.stdout


@pytest.fixture(scope="module")
def encoders() -> str:
    encoders = _ffmpeg_encoders()
    if "libx264rgb" not in encoders:
        pytest.skip("ffmpeg with libx264/libx264rgb is required for reference-video decode tests")
    return encoders


@pytest.fixture(scope="module")
def sources(tmp_path_factory, encoders) -> dict[str, str]:
    root = tmp_path_factory.mktemp("minimax_h3_reference_frames")
    paths: dict[str, str] = {}
    for name, (size, fps, seconds, audio, rotate) in SOURCES.items():
        path = root / f"{name}.mp4"
        command = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error"]
        command += ["-f", "lavfi", "-i", f"testsrc2=s={size}:r={fps}"]
        if audio:
            command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100"]
        command += ["-t", f"{seconds}", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
        if audio:
            command += ["-c:a", "aac"]
        if rotate is not None:
            command += ["-metadata:s:v:0", f"rotate={rotate}"]
        subprocess.run([*command, str(path)], check=True, timeout=60)
        paths[name] = str(path)
    # Video that starts 0.1 s after its audio: the first 24 fps frame lands at
    # pts 2, which ffmpeg's constant-rate output repeats down to time 0.
    base = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error"]
    video, audio = root / "late_video.h264.mp4", root / "late_video.m4a"
    subprocess.run(
        [*base, "-f", "lavfi", "-i", "testsrc2=s=320x256:r=30", "-t", "6", "-c:v", "libx264"]
        + ["-preset", "ultrafast", "-pix_fmt", "yuv420p", str(video)],
        check=True,
        timeout=60,
    )
    subprocess.run(
        [*base, "-f", "lavfi", "-i", "sine=sample_rate=44100", "-t", "6.2", "-c:a", "aac", str(audio)],
        check=True,
        timeout=60,
    )
    paths["late_video"] = str(root / "late_video.mp4")
    subprocess.run(
        [*base, "-itsoffset", "0.1", "-i", str(video), "-i", str(audio), "-map", "0:v", "-map", "1:a"]
        + ["-c", "copy", paths["late_video"]],
        check=True,
        timeout=60,
    )
    # Full-range (yuvj420p) video: ffmpeg's scaler warns about it.
    paths["full_range"] = str(root / "full_range.mp4")
    subprocess.run(
        [*base, "-f", "lavfi", "-i", "testsrc2=s=320x256:r=24", "-t", "4", "-c:v", "libx264", "-preset", "ultrafast"]
        + ["-pix_fmt", "yuvj420p", paths["full_range"]],
        check=True,
        timeout=60,
    )
    # H.264 Annex B segments with in-band SPS: concatenated, the decoder's
    # frame size or pixel format changes mid-stream and ffmpeg rebuilds the
    # filter graph there.
    segments = {}
    for segment, (size, pix_fmt, seconds) in {
        "a": ("320x256", "yuv420p", 3.1),
        "b": ("384x288", "yuv420p", 3.0),
        "c": ("320x256", "yuv444p", 3.0),
    }.items():
        segments[segment] = root / f"segment_{segment}.h264"
        subprocess.run(
            [*base, "-f", "lavfi", "-i", f"testsrc2=s={size}:r=30", "-t", f"{seconds}", "-c:v", "libx264"]
            + ["-preset", "ultrafast", "-pix_fmt", pix_fmt, "-bsf:v", "h264_mp4toannexb", "-f", "h264"]
            + [str(segments[segment])],
            check=True,
            timeout=60,
        )
    for name, (first, second), with_audio in (
        ("size_change", ("a", "b"), False),
        ("size_change_audio", ("a", "b"), True),
        ("format_change_audio", ("a", "c"), True),
    ):
        stream = root / f"{name}.h264"
        stream.write_bytes(segments[first].read_bytes() + segments[second].read_bytes())
        paths[name] = str(root / f"{name}.mp4")
        command = [*base, "-framerate", "30", "-i", str(stream)]
        if with_audio:
            command += ["-f", "lavfi", "-i", "sine=sample_rate=44100", "-map", "0:v", "-map", "1:a", "-c:a", "aac"]
            command += ["-t", "6.1"]
        subprocess.run([*command, "-c:v", "copy", paths[name]], check=True, timeout=60)
    return paths


@pytest.fixture(autouse=True)
def _cancel_frame_jobs(monkeypatch):
    """Stop every frame job a test leaves running, so one failure cannot leak ffmpeg children into later tests."""
    jobs = []
    init = reference_video.ReferenceVideoFrames.__init__

    def tracked_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        jobs.append(self)

    monkeypatch.setattr(reference_video.ReferenceVideoFrames, "__init__", tracked_init)
    yield
    for job in jobs:
        job.cancel()


def _ffmpeg_children() -> list[str]:
    names = []
    for child in psutil.Process().children(recursive=True):
        try:
            if child.name() in ("ffmpeg", "ffprobe"):
                names.append(f"{child.name()}:{child.pid}:{child.status()}")
        except psutil.NoSuchProcess:
            continue
    return names


def _assert_reaped(job: reference_video.ReferenceVideoFrames) -> None:
    assert job._processes
    assert all(process.returncode is not None for process in job._processes)
    for process in job._processes:
        assert all(pipe is None or pipe.closed for pipe in (process.stdout, process.stderr))
    assert not any(thread.is_alive() for thread in job._threads)
    assert _ffmpeg_children() == []


def _legacy_frames(source, workdir, *, width, height, frame_count, start, duration) -> torch.Tensor:
    workdir.mkdir()
    prepared = reference_video._transcode_reference_video(
        source,
        target_width=width,
        target_height=height,
        target_frame_count=frame_count,
        workdir=str(workdir),
        start_time_seconds=start,
        duration_seconds=duration,
    )
    return processing._frames_to_tensor(reference_video.load_video_frames(prepared))


@pytest.mark.parametrize(
    ("name", "start", "duration", "frame_count", "expected"),
    [
        ("s30_audio", 0.0, None, TARGET_FRAMES, TARGET_FRAMES),  # natural length, source longer than target
        ("s30_audio", 0.37, None, TARGET_FRAMES, TARGET_FRAMES),  # start offset
        # -t binds off a frame boundary: 4.52 * 24 = 108.48 keeps 108 frames.
        ("s30_audio", 0.0, 4.52, TARGET_FRAMES, 108),
        ("s30_audio", 0.0, None, 5, 5),  # fewer frames than workers
        # Multi-stream input whose first 24 fps frame lands after time 0:
        # constant-rate output repeats it, and -t counts the repeat.
        ("s30_audio", 0.37, 4.52, TARGET_FRAMES, 109),
        ("late_video", 0.0, None, TARGET_FRAMES, TARGET_FRAMES),
        ("late_video", 0.0, 4.52, TARGET_FRAMES, 109),
        ("late_video", 0.0, 3.03, TARGET_FRAMES, 73),
        ("s25_short", 0.0, None, TARGET_FRAMES, "short"),  # source shorter than target, no padding
        ("s25_short", 0.37, None, TARGET_FRAMES, "short"),
        ("s60_rotate", 0.0, None, TARGET_FRAMES, "short"),
        ("s60_rotate", 0.0, 4.52, TARGET_FRAMES, 108),
    ],
)
def test_direct_frames_match_prepared_video_round_trip(sources, tmp_path, name, start, duration, frame_count, expected):
    source = sources[name]
    meta = reference_video._probe_video(source)
    if duration is None:
        duration = float(meta["duration"]) - start
    kwargs = {"frame_count": frame_count, "start": start, "duration": duration}
    legacy = _legacy_frames(source, tmp_path / "legacy", width=WIDTH, height=HEIGHT, **kwargs)
    if expected == "short":
        assert legacy.shape[0] < frame_count
    else:
        assert legacy.shape[0] == expected
    for workers in WORKER_COUNTS:
        job = reference_video.start_reference_video_frames(
            source,
            target_width=WIDTH,
            target_height=HEIGHT,
            target_frame_count=frame_count,
            start_time_seconds=start,
            duration_seconds=duration,
            workers=workers,
            stream_count=meta["stream_count"],
        )
        assert job.workers == min(workers, frame_count)
        frames = job.result()
        assert not job.serial_fallback
        assert frames.dtype == torch.uint8 and frames.is_contiguous()
        assert tuple(frames.shape) == tuple(legacy.shape), workers
        assert torch.equal(frames, legacy), workers
        _assert_reaped(job)


@pytest.mark.parametrize(
    ("name", "start", "duration"),
    [
        ("size_change", 0.0, None),
        ("size_change", 0.0, 4.52),
        ("size_change_audio", 0.0, None),
        ("size_change_audio", 0.37, 4.52),
        ("format_change_audio", 0.0, None),
    ],
)
def test_mid_stream_frame_property_change_falls_back_to_one_process(sources, tmp_path, name, start, duration):
    # ffmpeg rebuilds the filter graph where the frame size or pixel format
    # changes, which a split decode cannot reproduce.
    source = sources[name]
    meta = reference_video._probe_video(source)
    if duration is None:
        duration = float(meta["duration"]) - start
    legacy = _legacy_frames(
        source,
        tmp_path / "legacy",
        width=WIDTH,
        height=HEIGHT,
        frame_count=TARGET_FRAMES,
        start=start,
        duration=duration,
    )
    for workers in (1, 2, 3, 8):
        job = reference_video.start_reference_video_frames(
            source,
            target_width=WIDTH,
            target_height=HEIGHT,
            target_frame_count=TARGET_FRAMES,
            start_time_seconds=start,
            duration_seconds=duration,
            workers=workers,
            stream_count=meta["stream_count"],
        )
        frames = job.result()
        assert job.serial_fallback == (workers > 1), workers
        assert tuple(frames.shape) == tuple(legacy.shape), workers
        assert torch.equal(frames, legacy), workers
        _assert_reaped(job)


def test_prepare_reference_videos_splits_workers_and_matches_transcode(sources, tmp_path):
    videos = [sources["s30_audio"], sources["s25_short"], sources["s60_rotate"]]
    legacy = reference_video.prepare_reference_videos(
        videos, target_frame_count=TARGET_FRAMES, workdir=str(tmp_path / "legacy"), start_time_seconds=0.37
    )
    direct = reference_video.prepare_reference_videos(
        videos,
        target_frame_count=TARGET_FRAMES,
        workdir=str(tmp_path / "direct"),
        start_time_seconds=0.37,
        decode_frames=True,
        frame_workers=8,
    )
    assert not (tmp_path / "direct").exists()
    assert [item["input_has_audio"] for item in direct] == [True, False, False]
    for old, new in zip(legacy, direct, strict=True):
        job = new.pop("frames")
        assert job.workers == 8 // len(videos)
        assert new == {**old, "prepared_path": None}
        frames = job.result()
        expected = processing._frames_to_tensor(reference_video.load_video_frames(old["prepared_path"]))
        assert torch.equal(frames, expected)
    assert _ffmpeg_children() == []


def test_prepare_reference_videos_validates_every_video_before_decoding(monkeypatch, tmp_path):
    metadata = {
        "width": 1280,
        "height": 720,
        "fps": 24.0,
        "frame_count": 240,
        "format_names": ("mp4",),
        "video_codec": "h264",
        "audio_codecs": (),
        "file_size": 1024,
    }
    durations = {"first.mp4": 10.0, "second.mp4": 5.02}
    started = []
    monkeypatch.setattr(
        reference_video, "_probe_video", lambda path: {**metadata, "duration": durations[os.path.basename(path)]}
    )
    monkeypatch.setattr(reference_video, "start_reference_video_frames", lambda *args, **kwargs: started.append(args))

    with pytest.raises(OmniClientError, match="15 seconds"):
        reference_video.prepare_reference_videos(
            [str(tmp_path / "first.mp4"), str(tmp_path / "second.mp4")],
            target_frame_count=TARGET_FRAMES,
            workdir=str(tmp_path / "work"),
            decode_frames=True,
            frame_workers=8,
        )
    assert started == []


@pytest.mark.parametrize("kind", ["missing", "corrupt"])
@pytest.mark.parametrize("workers", [1, 3])
def test_failed_decode_raises_called_process_error_and_reaps_processes(tmp_path, capfd, encoders, kind, workers):
    source = tmp_path / "bad.mp4"
    if kind == "corrupt":
        source.write_bytes(b"not a video\n" * 512)
    job = reference_video.start_reference_video_frames(
        str(source),
        target_width=WIDTH,
        target_height=HEIGHT,
        target_frame_count=TARGET_FRAMES,
        duration_seconds=4.0,
        workers=workers,
        stream_count=2,
    )
    assert job.workers == workers
    with pytest.raises(subprocess.CalledProcessError) as failure:
        job.result()
    _assert_reaped(job)
    job.cancel()
    # A split decode prints ffmpeg's errors once, as one process would, and keeps them on the error.
    printed = [line for line in capfd.readouterr().err.splitlines() if str(source) in line]
    assert len(printed) == 1 and "[error]" not in printed[0]
    if workers > 1:
        assert str(source).encode() in failure.value.stderr
    else:
        assert failure.value.stderr is None


def test_split_decode_prints_no_ffmpeg_warnings(sources, tmp_path, capfd):
    source = sources["full_range"]
    warning = "deprecated pixel format used"
    probe = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "warning", "-i", source, "-vf", f"scale={WIDTH}:{HEIGHT}", "-frames:v", "1"]
        + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        text=False,
        check=True,
        timeout=60,
    )
    assert warning.encode() in probe.stderr  # ffmpeg does warn about this source
    duration = float(reference_video._probe_video(source)["duration"])
    legacy = _legacy_frames(
        source, tmp_path / "legacy", width=WIDTH, height=HEIGHT, frame_count=TARGET_FRAMES, start=0.0, duration=duration
    )
    capfd.readouterr()
    job = reference_video.start_reference_video_frames(
        source,
        target_width=WIDTH,
        target_height=HEIGHT,
        target_frame_count=TARGET_FRAMES,
        duration_seconds=duration,
        workers=3,
        stream_count=1,
    )
    assert torch.equal(job.result(), legacy)
    # Like the transcode at -loglevel error, nothing below error level is printed.
    assert warning not in capfd.readouterr().err
    _assert_reaped(job)


def test_frame_protocol_error_is_the_cause_of_the_ffmpeg_failure(monkeypatch, sources):
    command = reference_video.ReferenceVideoFrames._command

    def without_frame_limit(self, *args, **kwargs):
        argv = command(self, *args, **kwargs)
        index = argv.index("-frames:v")
        return argv[:index] + argv[index + 2 :]

    monkeypatch.setattr(reference_video.ReferenceVideoFrames, "_command", without_frame_limit)
    for workers in (1, 3):
        job = reference_video.start_reference_video_frames(
            sources["s30_audio"],
            target_width=WIDTH,
            target_height=HEIGHT,
            target_frame_count=5,
            workers=workers,
            stream_count=2,
        )
        # The reader stops at the extra frame and closes the pipe, which fails ffmpeg too.
        with pytest.raises((subprocess.CalledProcessError, OmniClientError)) as failure:
            job.result()
        error = failure.value
        if isinstance(error, subprocess.CalledProcessError):
            error = error.__cause__
        assert isinstance(error, OmniClientError)
        assert "more reference video frames than requested" in str(error)
        _assert_reaped(job)


def test_single_process_command_is_the_transcode_with_raw_output(monkeypatch, tmp_path):
    transcodes, direct = [], []
    monkeypatch.setattr(reference_video.subprocess, "run", lambda command, **kwargs: transcodes.append(command))

    def popen(command, **kwargs):
        direct.append((command, kwargs.get("stderr")))
        raise OSError("not started")

    monkeypatch.setattr(reference_video.subprocess, "Popen", popen)
    options = {
        "target_width": 640,
        "target_height": 384,
        "target_frame_count": 124,
        "start_time_seconds": 0.37,
        "duration_seconds": 4.52,
    }
    reference_video._transcode_reference_video("in.mp4", workdir=str(tmp_path), **options)
    for workers in (1, 3):
        with pytest.raises(OSError, match="not started"):
            reference_video.start_reference_video_frames("in.mp4", workers=workers, stream_count=2, **options)
    transcode = transcodes[0]
    expected = [arg for arg in transcode[: transcode.index("-metadata:s:v:0")] if arg != "-y"]
    assert direct[0] == ([*expected, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], None)
    split, split_stderr = direct[1]
    assert split[:8] == ["ffmpeg", "-nostdin", "-loglevel", "level+warning", "-filter_threads", "1"] + [
        "-reinit_filter",
        "0",
    ]
    assert split_stderr is subprocess.PIPE


def test_start_failure_stops_processes_already_started(monkeypatch, sources):
    real_popen = subprocess.Popen
    started = []

    def popen(*args, **kwargs):
        if len(started) == 2:
            raise OSError("cannot start ffmpeg")
        started.append(real_popen(*args, **kwargs))
        return started[-1]

    monkeypatch.setattr(reference_video.subprocess, "Popen", popen)
    with pytest.raises(OSError, match="cannot start ffmpeg"):
        reference_video.start_reference_video_frames(
            sources["s30_audio"],
            target_width=WIDTH,
            target_height=HEIGHT,
            target_frame_count=TARGET_FRAMES,
            duration_seconds=5.0,
            workers=8,
            stream_count=2,
        )
    assert len(started) == 2
    assert all(process.returncode is not None and process.stdout.closed for process in started)
    assert _ffmpeg_children() == []


def test_cancel_stops_every_process(sources):
    job = reference_video.start_reference_video_frames(
        sources["s30_audio"],
        target_width=WIDTH,
        target_height=HEIGHT,
        target_frame_count=TARGET_FRAMES,
        duration_seconds=5.0,
        workers=3,
        stream_count=2,
    )
    assert job.workers == 3
    job.cancel()
    _assert_reaped(job)
    job.cancel()
    with pytest.raises(RuntimeError, match="already consumed"):
        job.result()


def test_unknown_stream_count_decodes_in_one_process(sources):
    job = reference_video.start_reference_video_frames(
        sources["s30_audio"],
        target_width=WIDTH,
        target_height=HEIGHT,
        target_frame_count=TARGET_FRAMES,
        duration_seconds=5.0,
        workers=8,
    )
    assert job.workers == 1
    job.cancel()
    _assert_reaped(job)


def test_reference_frame_workers_env(monkeypatch):
    name = reference_video.MINIMAX_H3_REFERENCE_DECODE_WORKERS_ENV
    monkeypatch.delenv(name, raising=False)
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    assert reference_video.reference_frame_workers() == min(8, max(1, cpus // 8))
    for raw, expected in (("0", 0), ("1", 1), (" 12 ", 12)):
        monkeypatch.setenv(name, raw)
        assert reference_video.reference_frame_workers() == expected
    for raw in ("-1", "eight"):
        monkeypatch.setenv(name, raw)
        with pytest.raises(ValueError, match=name):
            reference_video.reference_frame_workers()


def test_legacy_transcode_command_is_unchanged(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(reference_video.subprocess, "run", lambda command, **kwargs: commands.append(command))
    common = {"target_width": 640, "target_height": 384, "target_frame_count": 124, "workdir": str(tmp_path)}
    reference_video._transcode_reference_video("in.mp4", start_time_seconds=0.37, duration_seconds=4.52, **common)
    reference_video._transcode_reference_video("in.mp4", pad_last_frame=True, cover=True, **common)
    head = "ffmpeg -nostdin -y -loglevel error -ss"
    tail = f"-metadata:s:v:0 rotate=0 -c:v libx264rgb -crf 0 -preset veryfast -pix_fmt rgb24 {tmp_path}/prepared.mp4"
    assert [" ".join(command) for command in commands] == [
        f"{head} 0.370000 -i in.mp4 -map 0:v:0 -an -vf fps=24,scale=640:384:flags=lanczos,setsar=1 "
        f"-t 4.520000 -frames:v 124 {tail}",
        f"{head} 0.000000 -i in.mp4 -map 0:v:0 -an -vf fps=24,"
        "scale=640:384:force_original_aspect_ratio=increase:flags=bilinear,crop=640:384,setsar=1,"
        f"tpad=stop_mode=clone:stop=-1 -frames:v 124 {tail}",
    ]


@pytest.fixture(scope="module")
def reference_mp3(tmp_path_factory, encoders):
    if "libmp3lame" not in encoders:
        pytest.skip("ffmpeg with libmp3lame is required for reference-audio tests")
    root = tmp_path_factory.mktemp("minimax_h3_reference_audio")
    paths = {}
    for name, seconds in (("valid", 3.0), ("too_short", 1.0), ("too_long", 16.0)):
        path = root / f"{name}.mp3"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi", "-i"]
            + ["sine=frequency=330:sample_rate=44100", "-t", f"{seconds}", "-c:a", "libmp3lame", str(path)],
            check=True,
            timeout=60,
        )
        paths[name] = str(path)
    return paths


def _sha(value) -> str:
    array = value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _digest(prepared) -> dict:
    media = prepared.media
    return {
        "shape": (media.height, media.width, media.num_frames, media.latent_t, media.audio_t),
        "images": [(tuple(image.shape), _sha(image)) for image in media.images],
        "videos": [(tuple(video.shape), video.dtype, _sha(video)) for video in media.videos],
        "video_audios": [
            None if item is None else (tuple(item[0].shape), item[1], _sha(item[0])) for item in media.video_audios
        ],
        "audios": [(tuple(waveform.shape), rate, _sha(waveform)) for waveform, rate in media.audios],
        "qwen_videos": [(frames.shape, _sha(frames), meta) for frames, meta in prepared.qwen_videos],
        "pil_images": [(image.size, image.mode, _sha(np.asarray(image))) for image in prepared.images],
        "video_timestamps": prepared.video_timestamps,
        "condition_labels": prepared.condition_labels,
    }


def _images() -> list[Image.Image]:
    rng = np.random.default_rng(0)
    return [
        Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
        for width, height in ((300, 280), (520, 260))
    ]


def _prepare(videos, audio, *, images=True, start=None, task="ref2va", audio_mode=None):
    multi_modal_data = {"video": videos, "audio": audio}
    if images:
        multi_modal_data["image"] = _images()
    extra_args = {"task": task, "duration": 4.0, "start_time_seconds": start}
    if audio_mode is not None:
        extra_args["audio_mode"] = audio_mode
    return processing.prepare_encoder_inputs(
        {"prompt": "reference", "multi_modal_data": multi_modal_data},
        OmniDiffusionSamplingParams(height=256, width=384, extra_args=extra_args),
    )


def _count_audio_work(monkeypatch) -> dict[str, int]:
    """Count reference-audio probes and decodes, on every path that makes them."""
    calls = {"probe": 0, "load": 0}
    probe, load = reference_video._probe_audio, processing.load_audio_file

    def counted_probe(path):
        calls["probe"] += 1
        return probe(path)

    def counted_load(path):
        calls["load"] += 1
        return load(path)

    monkeypatch.setattr(reference_video, "_probe_audio", counted_probe)
    monkeypatch.setattr(processing, "load_audio_file", counted_load)
    return calls


def test_encoder_inputs_are_identical_to_the_legacy_path(monkeypatch, sources, reference_mp3):
    name = reference_video.MINIMAX_H3_REFERENCE_DECODE_WORKERS_ENV
    legacy_loads, prefetches, resizes = [], [], []
    load_video_frames = processing.load_video_frames
    monkeypatch.setattr(
        processing, "load_video_frames", lambda path: legacy_loads.append(path) or load_video_frames(path)
    )
    start = processing._ReferenceAudioPrefetch.start
    monkeypatch.setattr(
        processing._ReferenceAudioPrefetch,
        "start",
        lambda self, values: prefetches.append(list(values)) or start(self, values),
    )
    prepare_images = processing._prepare_encoder_images
    monkeypatch.setattr(
        processing,
        "_prepare_encoder_images",
        lambda *args, **kwargs: resizes.append(kwargs["parallel"]) or prepare_images(*args, **kwargs),
    )
    digests = {}
    for raw in ("0", "1", "3", None):
        if raw is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, raw)
        legacy_loads.clear()
        prefetches.clear()
        resizes.clear()
        digests[raw] = _digest(_prepare([sources["s30_audio"]], reference_mp3["valid"], start=0.37))
        # 0 is the whole serial path: no direct decode, no audio prefetch, no parallel resize.
        assert len(legacy_loads) == (1 if raw == "0" else 0)
        assert prefetches == ([] if raw == "0" else [[reference_mp3["valid"]]]), raw
        assert resizes == [raw != "0"], raw
        assert _ffmpeg_children() == []
    assert digests["0"]["video_audios"][0] is not None
    assert digests["0"]["condition_labels"] == [("image", 1), ("image", 2), ("audio", 1), ("video", 1), ("audio", 2)]
    for raw in ("1", "3", None):
        assert digests[raw] == digests["0"], raw


def test_lock_source_audio_is_neither_probed_nor_prefetched(monkeypatch, sources, reference_mp3):
    name = reference_video.MINIMAX_H3_REFERENCE_DECODE_WORKERS_ENV
    digests = {}
    for raw in ("0", "8"):
        monkeypatch.setenv(name, raw)
        calls = _count_audio_work(monkeypatch)
        prepared = _prepare([sources["s30_audio"]], reference_mp3["valid"], audio_mode="lock_source")
        assert calls == {"probe": 0, "load": 1}, raw
        digests[raw] = _digest(prepared)
    assert digests["8"] == digests["0"]


@pytest.mark.parametrize("workers", ["0", "8"])
@pytest.mark.parametrize(
    ("audio", "match"),
    [
        ("too_short", r"reference audio 0 duration must be in \[2, 15\] seconds, got 1\.0"),
        ("too_long", r"reference audio 0 duration must be in \[2, 15\] seconds, got 16\.0"),
    ],
)
def test_invalid_reference_audio_error_is_unchanged_and_is_never_decoded(
    monkeypatch, sources, reference_mp3, workers, audio, match
):
    monkeypatch.setenv(reference_video.MINIMAX_H3_REFERENCE_DECODE_WORKERS_ENV, workers)
    calls = _count_audio_work(monkeypatch)
    with pytest.raises(OmniClientError, match=match):
        _prepare([sources["s25_short"]], reference_mp3[audio], images=False)
    assert calls == {"probe": 1, "load": 0}
    assert _ffmpeg_children() == []


@pytest.mark.parametrize(
    ("task", "audio_count", "match"),
    [
        ("t2va", 1, "t2va does not accept image, video, or audio conditions"),
        ("ref2va", 4, "ref2va accepts at most 3 standalone audio references"),
    ],
)
def test_rejected_requests_touch_no_reference_audio(monkeypatch, reference_mp3, task, audio_count, match):
    monkeypatch.delenv(reference_video.MINIMAX_H3_REFERENCE_DECODE_WORKERS_ENV, raising=False)
    calls = _count_audio_work(monkeypatch)
    with pytest.raises(OmniClientError, match=match):
        _prepare([], [reference_mp3["valid"]] * audio_count, images=task == "ref2va", task=task)
    assert calls == {"probe": 0, "load": 0}


def test_unconsumed_frame_jobs_are_cancelled_on_error(monkeypatch, sources, reference_mp3):
    monkeypatch.setenv(reference_video.MINIMAX_H3_REFERENCE_DECODE_WORKERS_ENV, "2")
    jobs = []
    start = reference_video.start_reference_video_frames

    def record(*args, **kwargs):
        jobs.append(start(*args, **kwargs))
        return jobs[-1]

    def fail_sampling(*args, **kwargs):
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(reference_video, "start_reference_video_frames", record)
    monkeypatch.setattr(processing, "sample_reference_video_frames", fail_sampling)
    with pytest.raises(RuntimeError, match="sampling failed"):
        _prepare([sources["s30_audio"], sources["s60_rotate"]], reference_mp3["valid"], images=False)
    assert [job.workers for job in jobs] == [1, 1]
    for job in jobs:
        _assert_reaped(job)

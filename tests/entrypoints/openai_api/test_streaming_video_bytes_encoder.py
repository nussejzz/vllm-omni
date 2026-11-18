# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The streaming response encoder must reproduce ``_encode_video_bytes`` byte for byte."""

import threading

import numpy as np
import pytest

from vllm_omni.entrypoints.openai import video_api_utils
from vllm_omni.entrypoints.openai.video_api_utils import (
    StreamingVideoBytesEncoder,
    _encode_video_bytes,
    _PlanarFrameConverter,
    streaming_response_frame_path,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# The serving defaults: ``threads=0`` lets libx264 size its threads and
# slices from the CPU count, as it does in the API process.
SERVING_OPTIONS = {"preset": "ultrafast", "threads": "0"}
SAMPLE_RATE = 32000


def _video(frames=9, height=128, width=160, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = np.stack([(x + y) % 256, (2 * x) % 256, (3 * y) % 256], axis=-1)
    video = np.stack([(base + 7 * index) % 256 for index in range(frames)]).astype(np.uint8)
    noise = rng.integers(0, 16, video.shape, dtype=np.uint8)
    return np.ascontiguousarray(video + noise)


def _audio(channels, samples=SAMPLE_RATE // 3, seed=1):
    if channels is None:
        return None
    rng = np.random.default_rng(seed)
    waveform = rng.standard_normal((1, channels, samples)).astype(np.float32) * 0.1
    # The worker hands the API a (1, C, N) array; mono reaches it squeezed.
    return waveform if channels > 1 else waveform[0, 0]


def _api_bytes(video, audio, frame_path, fps=24):
    converter = _PlanarFrameConverter(max_workers=8)
    try:
        return _encode_video_bytes(
            video,
            fps=fps,
            video_codec_options=SERVING_OPTIONS,
            frame_converter=converter,
            enable_borrowed_frames=frame_path == "borrowed_rgb",
            **({} if audio is None else {"audio": audio, "audio_sample_rate": SAMPLE_RATE}),
        )
    finally:
        converter.shutdown()


def _encoder(video, channels, frame_path, **kwargs):
    return StreamingVideoBytesEncoder(
        width=video.shape[2],
        height=video.shape[1],
        fps=kwargs.pop("fps", 24),
        frame_path=frame_path,
        audio_channels=channels,
        audio_sample_rate=SAMPLE_RATE,
        video_codec_options=SERVING_OPTIONS,
        **kwargs,
    )


@pytest.mark.parametrize("frame_path", ["direct_planar", "borrowed_rgb"])
@pytest.mark.parametrize("channels", [None, 1, 2])
@pytest.mark.parametrize("chunks", [(9,), (1, 4, 4), (1,) * 9], ids=["whole", "clips", "frames"])
def test_streamed_chunks_reproduce_the_response_encoding(frame_path, channels, chunks):
    video = _video()
    audio = _audio(channels)
    encoder = _encoder(video, channels, frame_path)
    start = 0
    for index, length in enumerate(chunks):
        chunk = video[start : start + length]
        start += length
        # Mix ready arrays with chunks that land only when the muxer asks.
        encoder.push(chunk if index % 2 else (lambda chunk=chunk: np.ascontiguousarray(chunk)))
    streamed = encoder.finish(audio)

    assert streamed == _api_bytes(video, audio, frame_path)
    assert encoder.frame_count == len(video)


def test_requested_frame_rate_reaches_the_stream():
    video = _video(frames=4)
    encoder = _encoder(video, None, "direct_planar", fps=30.0)
    encoder.push(video)

    assert encoder.finish() == _api_bytes(video, None, "direct_planar", fps=30.0)


@pytest.mark.parametrize("workers", [1, 3])
def test_conversion_workers_do_not_change_the_bytes(workers):
    video = _video()
    audio = _audio(2)
    encoder = _encoder(video, 2, "direct_planar", frame_conversion_workers=workers)
    encoder.push(video[:5])
    encoder.push(video[5:])

    assert encoder.finish(audio) == _api_bytes(video, audio, "direct_planar")


def test_chunks_are_resolved_and_released_in_order_on_the_muxing_thread():
    video = _video(frames=6)
    events = []
    main = threading.get_ident()
    encoder = _encoder(video, None, "direct_planar", max_pending=1)

    def landed(index):
        def wait():
            events.append(("wait", index, threading.get_ident() != main))
            return video[2 * index : 2 * index + 2]

        return wait

    for index in range(3):
        encoder.push(landed(index), on_consumed=lambda index=index: events.append(("release", index, None)))
    encoder.finish()

    assert events == [item for index in range(3) for item in (("wait", index, True), ("release", index, None))]


def test_audio_that_does_not_match_the_declared_track_fails_and_releases_every_chunk():
    video = _video(frames=4)
    released = []
    encoder = _encoder(video, 2, "direct_planar")
    encoder.push(video[:2], on_consumed=lambda: released.append(0))
    encoder.push(video[2:], on_consumed=lambda: released.append(1))

    with pytest.raises(ValueError, match="declared 2-channel track"):
        encoder.finish(_audio(1))

    assert sorted(released) == [0, 1]
    encoder.abort()


def test_close_returns_before_the_mp4_is_complete():
    video = _video()
    audio = _audio(2)
    gate = threading.Event()
    encoder = _encoder(video, 2, "direct_planar")
    encoder.push(video[:4])
    encoder.push(lambda: gate.wait() and video[4:])

    encoder.close(audio)
    assert encoder._thread.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        encoder.close(audio)
    gate.set()

    assert encoder.result() == _api_bytes(video, audio, "direct_planar")


def test_result_requires_close():
    video = _video(frames=1)
    encoder = _encoder(video, None, "direct_planar")

    with pytest.raises(RuntimeError, match="close"):
        encoder.result()
    encoder.abort()


def test_missing_audio_for_a_declared_track_fails():
    video = _video(frames=2)
    encoder = _encoder(video, 2, "direct_planar")
    encoder.push(video)

    with pytest.raises(ValueError, match="exactly when audio_channels"):
        encoder.finish()


def test_abort_releases_queued_chunks_and_rejects_later_pushes():
    video = _video(frames=4)
    gate = threading.Event()
    released = []
    encoder = _encoder(video, None, "direct_planar", max_pending=4)

    def blocked():
        gate.wait()
        return video[:2]

    encoder.push(blocked, on_consumed=lambda: released.append("blocked"))
    encoder.push(video[2:], on_consumed=lambda: released.append("queued"))
    abort = threading.Thread(target=encoder.abort)
    abort.start()
    gate.set()
    abort.join(timeout=30)

    assert not abort.is_alive()
    assert sorted(released) == ["blocked", "queued"]
    with pytest.raises(RuntimeError, match="aborted|closed"):
        encoder.push(video[:1], on_consumed=lambda: released.append("late"))
    assert released.count("late") == 1


@pytest.mark.parametrize(
    "chunk",
    [
        np.zeros((1, 128, 160, 3), dtype=np.float32),
        np.zeros((1, 128, 160, 4), dtype=np.uint8),
        np.zeros((1, 64, 160, 3), dtype=np.uint8),
        np.zeros((1, 128, 160, 6), dtype=np.uint8)[..., ::2],
    ],
    ids=["dtype", "channels", "size", "strided"],
)
def test_invalid_chunks_are_rejected_and_released(chunk):
    video = _video(frames=1)
    released = []
    encoder = _encoder(video, None, "direct_planar")

    with pytest.raises(ValueError):
        encoder.push(chunk, on_consumed=lambda: released.append(True))

    assert released == [True]
    encoder.abort()


def test_encoder_failures_surface_and_release_chunks():
    video = _video(frames=2)
    released = []
    encoder = StreamingVideoBytesEncoder(
        width=video.shape[2],
        height=video.shape[1],
        fps=24,
        video_codec_options={"preset": "not-a-preset"},
    )
    encoder.push(video, on_consumed=lambda: released.append(True))

    with pytest.raises(Exception):
        encoder.finish()

    assert released == [True]


@pytest.mark.parametrize(
    ("workers", "borrowed", "expected"),
    [(8, False, "direct_planar"), (8, True, "borrowed_rgb"), (1, True, "borrowed_rgb"), (1, False, None)],
)
def test_streaming_frame_path_names_the_route_encode_video_bytes_takes(workers, borrowed, expected, monkeypatch):
    converter = _PlanarFrameConverter(max_workers=workers)
    routes = []
    monkeypatch.setattr(
        video_api_utils,
        "_log_video_encoding_path",
        lambda **kwargs: routes.append(kwargs["selected_path"]),
    )
    try:
        assert streaming_response_frame_path(frame_converter=converter, enable_borrowed_frames=borrowed) == expected
        _encode_video_bytes(
            _video(frames=1),
            fps=24,
            video_codec_options=SERVING_OPTIONS,
            frame_converter=converter,
            enable_borrowed_frames=borrowed,
        )
    finally:
        converter.shutdown()
    assert routes == [expected or "legacy_fallback"]

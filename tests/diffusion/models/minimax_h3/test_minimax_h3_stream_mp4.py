# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for streaming MiniMax-H3's response MP4 out of the VAE decode.

The streamed MP4 must be byte-identical to what the video API encodes from
the frames the raw-frame path returns, and every mismatch must fall back to
that path instead of failing the request.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3 import stream_mp4
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    _minimax_h3_post_process,
    _MiniMaxH3EncodedOutput,
    _prepare_minimax_h3_video_output,
)
from vllm_omni.diffusion.models.minimax_h3.stream_mp4 import (
    MINIMAX_H3_STREAM_MP4_ENV,
    MiniMaxH3ResponseMP4Stream,
    resolve_stream_mp4_request,
)
from vllm_omni.diffusion.models.minimax_h3.vae_temporal import (
    _decode_temporal_streaming_uint8,
    publish_decoded_frames,
)
from vllm_omni.entrypoints.openai.video_api_utils import (
    STREAMING_RESPONSE_ENCODING_KEY,
    _encode_video_bytes,
    _PlanarFrameConverter,
    usable_cpu_count,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

SERVING_OPTIONS = {"preset": "ultrafast", "threads": "0"}
NUM_CHUNKS = 4


class _DecodeModel:
    """The checkpoint's streaming decode contract: overlapping clips, blended tails."""

    def __init__(self):
        self.tokens_chunk_size = 2
        self.vae_ratio_t = 1
        self.token_overlap = 1
        self.token_drop = 1
        self.frame_pre_padding = 0
        self.frame_overlap = 1
        self.processor = SimpleNamespace(transform_rev=SimpleNamespace(mean=(-1.0,) * 3, std=(2.0,) * 3))

    def _adaptive_decode(self, clip_z):
        return torch.sin(clip_z * 3.0)

    def blend(self, a, b, overlap, dim=-3):
        return (a + b) / 2.0

    def _decode_temporal_output_frame_plan(self, z, z_head, z_tail, num_chunks, pad_tokens):
        total = num_chunks * self.tokens_chunk_size * self.vae_ratio_t + self.frame_overlap
        return total, int(pad_tokens), total - int(pad_tokens)


def _latent(height=64, width=96, seed=0):
    generator = torch.Generator().manual_seed(seed)
    tokens = NUM_CHUNKS * 2 + 1
    return torch.randn((1, 3, tokens, height, width), generator=generator)


def _run_decode(z):
    """The fork under whatever sink the caller's scope installed."""
    return _decode_temporal_streaming_uint8(_DecodeModel(), z.clone(), None, None, NUM_CHUNKS, 0, None)


def _decode(z, sink=None):
    with publish_decoded_frames(sink):
        return _run_decode(z)


def _spec(frame_path="direct_planar", **overrides):
    spec = {
        "frame_path": frame_path,
        "video_codec_options": SERVING_OPTIONS,
        "affinity_cpus": usable_cpu_count(),
        "fps": None,
    }
    spec.update(overrides)
    return spec


def _stream(spec=None, *, height, width, channels=2):
    return MiniMaxH3ResponseMP4Stream(
        spec or _spec(),
        height=height,
        width=width,
        fps=24,
        audio_channels=channels,
        audio_sample_rate=32000,
        device=torch.device("cpu"),
    )


def _api_bytes(video, audio, *, frame_path="direct_planar"):
    """What the API encodes from the raw-frame path's output."""
    result = _minimax_h3_post_process((_prepare_minimax_h3_video_output(video), audio))
    converter = _PlanarFrameConverter(max_workers=8)
    try:
        return _encode_video_bytes(
            result["video"][0],
            fps=result["fps"],
            audio=result["audio"],
            audio_sample_rate=result["audio_sample_rate"],
            video_codec_options=SERVING_OPTIONS,
            frame_converter=converter,
            enable_borrowed_frames=frame_path == "borrowed_rgb",
        )
    finally:
        converter.shutdown()


def _audio(channels=2, samples=16000, seed=3):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((1, channels, samples), generator=generator) * 0.1


def test_decode_publishes_every_final_frame_range_in_order():
    z = _latent(height=8, width=8)
    published = []

    def sink(dec, start, end):
        published.append((dec, start, end, dec[:, :, start:end].clone()))

    out = _decode(z, sink)

    assert torch.equal(out, _decode(z))
    assert [(start, end) for _, start, end, _ in published] == [(0, 2), (2, 4), (4, 6), (6, 8), (8, 9)]
    for dec, start, end, frames in published:
        assert dec is out
        assert torch.equal(frames, out[:, :, start:end])


@pytest.mark.parametrize("frame_path", ["direct_planar", "borrowed_rgb"])
@pytest.mark.parametrize("crop", [(64, 96), (64, 80)], ids=["canvas", "cropped"])
def test_streamed_mp4_is_the_api_encoding_of_the_returned_frames(frame_path, crop):
    height, width = crop
    z = _latent()
    stream = _stream(_spec(frame_path), height=height, width=width)
    video = _decode(z, stream)[..., :height, :width].contiguous()
    audio = _audio()

    encoded = stream.finish(video, audio)

    assert encoded is not None
    data, frame_count = encoded
    assert frame_count == video.shape[2]
    assert data == _api_bytes(video, audio, frame_path=frame_path)


def _assert_falls_back(stream, video, audio):
    assert stream.finish(video, audio) is None
    assert not stream._encoder._thread.is_alive()
    assert stream._ring.slots_in_use == 0


def test_stream_falls_back_when_the_returned_video_is_not_what_was_streamed():
    z = _latent()
    stream = _stream(height=64, width=96)
    video = _decode(z, stream).clone()
    video[0, 0, 3, 5, 7] ^= 1

    _assert_falls_back(stream, video, _audio())


def test_stream_falls_back_when_the_decode_never_published():
    stream = _stream(height=64, width=96)
    video = _decode(_latent())

    _assert_falls_back(stream, video, _audio())


def test_stream_falls_back_on_out_of_order_frames_without_raising_into_the_decode():
    z = _latent()
    stream = _stream(height=64, width=96)
    out = _decode(z, stream)
    stream(out, 2, 4)

    _assert_falls_back(stream, out, _audio())


def test_stream_falls_back_when_the_waveform_does_not_fit_the_declared_track():
    z = _latent()
    stream = _stream(height=64, width=96, channels=2)
    video = _decode(z, stream)

    _assert_falls_back(stream, video, _audio(channels=1))


def test_abort_stops_the_encoder_and_returns_every_slot():
    stream = _stream(height=64, width=96)
    out = _decode(_latent(), stream)
    del out

    stream.abort()

    assert not stream._encoder._thread.is_alive()
    assert stream._ring.slots_in_use == 0


def test_resolve_stream_mp4_request(monkeypatch):
    monkeypatch.delenv(MINIMAX_H3_STREAM_MP4_ENV, raising=False)
    spec = _spec()
    assert resolve_stream_mp4_request({STREAMING_RESPONSE_ENCODING_KEY: spec}) == spec
    assert resolve_stream_mp4_request(None) is None
    assert resolve_stream_mp4_request({"preencode_mp4": True}) is None
    assert resolve_stream_mp4_request({STREAMING_RESPONSE_ENCODING_KEY: _spec("legacy_fallback")}) is None
    assert resolve_stream_mp4_request({STREAMING_RESPONSE_ENCODING_KEY: _spec(video_codec_options="fast")}) is None
    # libx264 would size its threads, and so its slices, differently.
    mismatch = _spec(affinity_cpus=usable_cpu_count() + 1)
    assert resolve_stream_mp4_request({STREAMING_RESPONSE_ENCODING_KEY: mismatch}) is None
    monkeypatch.setenv(MINIMAX_H3_STREAM_MP4_ENV, "0")
    assert resolve_stream_mp4_request({STREAMING_RESPONSE_ENCODING_KEY: spec}) is None


@pytest.mark.parametrize(("value", "expected"), [("", True), ("1", True), ("0", False), ("off", False)])
def test_stream_mp4_env_values(monkeypatch, value, expected):
    monkeypatch.setenv(MINIMAX_H3_STREAM_MP4_ENV, value)
    assert stream_mp4.stream_mp4_enabled() is expected


def test_stream_mp4_env_rejects_unknown_values(monkeypatch):
    monkeypatch.setenv(MINIMAX_H3_STREAM_MP4_ENV, "2")
    with pytest.raises(ValueError, match=MINIMAX_H3_STREAM_MP4_ENV):
        stream_mp4.stream_mp4_enabled()


def _forward(extra_args, *, num_outputs=1, height=64, width=80):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    z = _latent()
    audio = _audio()
    releases = []

    def decode(video_latent, audio_latent, *, height, width):
        return _run_decode(z)[..., :height, :width].contiguous(), audio.clone()

    context = {"num_outputs": num_outputs, "seed": 41, "height": height, "width": width, "preencode_mp4": False}
    pipeline = SimpleNamespace(
        load_vae_decoder=True,
        device=torch.device("cpu"),
        od_config=SimpleNamespace(),
        _prepare_request_inputs=lambda *args, **kwargs: context,
        _denoise_kwargs=lambda ctx: {},
        diffuse=lambda **kwargs: (torch.zeros(1), torch.zeros(1)),
        decode=decode,
        _offload_model_cpu_stage_output=lambda value: value,
        _release_stage_cache=lambda: releases.append(True),
    )
    request = SimpleNamespace(prompts=["a prompt"], sampling_params=SimpleNamespace(extra_args=extra_args))
    output = MiniMaxH3Pipeline.forward(pipeline, request)
    expected = _api_bytes(_decode(z)[..., :height, :width].contiguous(), audio)
    return output, expected, releases


def test_forward_returns_the_streamed_mp4(monkeypatch):
    monkeypatch.delenv(MINIMAX_H3_STREAM_MP4_ENV, raising=False)

    output, expected, releases = _forward({STREAMING_RESPONSE_ENCODING_KEY: _spec()})

    assert isinstance(output.output, _MiniMaxH3EncodedOutput)
    assert output.output.video == expected
    assert output.output.frame_counts == [NUM_CHUNKS * 2 + 1]
    assert output.video_output_index is None
    assert releases == [True]


@pytest.mark.parametrize(
    ("env", "extra_args", "num_outputs"),
    [
        ("0", {STREAMING_RESPONSE_ENCODING_KEY: _spec()}, 1),
        ("1", {}, 1),
        ("1", {STREAMING_RESPONSE_ENCODING_KEY: _spec()}, 2),
    ],
    ids=["switched-off", "not-requested", "several-outputs"],
)
def test_forward_keeps_the_raw_frame_path(monkeypatch, env, extra_args, num_outputs):
    monkeypatch.setenv(MINIMAX_H3_STREAM_MP4_ENV, env)

    output, expected, _ = _forward(extra_args, num_outputs=num_outputs)

    video, audio = output.output
    assert isinstance(video, torch.Tensor) and video.dtype == torch.uint8
    result = _minimax_h3_post_process((video[:1], audio[:1]))
    assert (
        _encode_video_bytes(
            result["video"][0],
            fps=result["fps"],
            audio=result["audio"],
            audio_sample_rate=result["audio_sample_rate"],
            video_codec_options=SERVING_OPTIONS,
            frame_converter=_PlanarFrameConverter(max_workers=8),
        )
        == expected
    )
    assert np.asarray(video).shape[0] == num_outputs

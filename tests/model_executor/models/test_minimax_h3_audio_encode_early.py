# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Rank 0's early reference-audio WVAE encode must hand encode_media the serial path's rows."""

import hashlib
import shutil
import subprocess
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.minimax_h3 import encoder_processing as processing
from vllm_omni.model_executor.models.minimax_h3 import reference_video
from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderMediaInput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

WORKERS_ENV = reference_video.MINIMAX_H3_REFERENCE_DECODE_WORKERS_ENV


class FakeAudioVAE:
    """40 latents per second like the WVAE; every row depends on every input sample."""

    def __init__(self, *, fail: int = 0, events: list[str] | None = None) -> None:
        self.calls: list[tuple[torch.Tensor, int]] = []
        self.threads: list[int] = []
        self.idents: list[int] = []
        self.fail = fail
        self.events = events

    def encode_waveform(self, waveform: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, int]:
        self.calls.append((waveform, sample_rate))
        self.threads.append(torch.get_num_threads())
        self.idents.append(threading.get_ident())
        if self.events is not None:
            self.events.append("encode")
        if self.fail:
            self.fail -= 1
            raise RuntimeError("audio WVAE failed")
        length = round(waveform.shape[-1] / sample_rate * 40)
        flat = waveform.reshape(-1).double()
        value = float((flat * torch.arange(1, flat.numel() + 1, dtype=torch.float64)).sum())
        return torch.full((2 * length, 32), value, dtype=torch.float32), length


class FakeVideoVAE:
    def encode_image(self, image):
        return torch.ones(1, 96)

    def encode_video(self, frames):
        shape = (1 + (frames.shape[0] - 1) // 4, frames.shape[1] // 16, frames.shape[2] // 16)
        return torch.full((shape[0] * (shape[1] // 2) * (shape[2] // 2), 96), float(frames.mean())), shape


def _recording_scope():
    entered = []

    @contextmanager
    def scope(component):
        entered.append(component)
        yield

    return scope, entered


def _media() -> MiniMaxH3EncoderMediaInput:
    rate = 16000
    t = torch.arange(3 * rate, dtype=torch.float32)
    # 60 frames are 2.5 s: the 3 s standalone reference is cropped to a strided view.
    return MiniMaxH3EncoderMediaInput(
        task="ref2va",
        height=32,
        width=32,
        num_frames=60,
        latent_t=16,
        audio_t=100,
        videos=(torch.zeros(5, 32, 32, 3, dtype=torch.uint8),),
        video_audios=((torch.stack([torch.sin(t / 7), torch.cos(t / 11)]), rate),),
        audios=((torch.stack([torch.sin(t / 5), torch.cos(t / 3)]), rate),),
    )


def _audio_inputs(media):
    return processing._effective_audio_inputs(
        media.video_audios, media.audios, max_standalone_seconds=media.num_frames / processing.MINIMAX_H3_FPS
    )


def _assert_same_conditioning(actual, expected) -> None:
    assert torch.equal(actual.audio_condition, expected.audio_condition)
    assert actual.audio_condition_lengths == expected.audio_condition_lengths
    assert actual.ref_blocks == expected.ref_blocks
    assert torch.equal(actual.visual_condition, expected.visual_condition)
    assert actual.visual_condition_shapes == expected.visual_condition_shapes


def test_audio_encode_early_env(monkeypatch):
    name = processing.MINIMAX_H3_AUDIO_ENCODE_EARLY_ENV
    assert name == "VLLM_OMNI_H3_AUDIO_ENCODE_EARLY"
    monkeypatch.delenv(name, raising=False)
    assert processing.audio_encode_early() is True
    for raw, expected in (("1", True), ("0", False), (" 0 ", False)):
        monkeypatch.setenv(name, raw)
        assert processing.audio_encode_early() is expected
    for raw in ("2", "off", "true"):
        monkeypatch.setenv(name, raw)
        with pytest.raises(ValueError, match=name):
            processing.audio_encode_early()


def test_take_returns_rows_only_for_the_exact_waveform():
    vae = FakeAudioVAE()
    scope, entered = _recording_scope()
    encodings = processing.ReferenceAudioEncodings(vae, component_scope=scope)
    encodings.encode([])
    assert entered == [] and len(encodings) == 0  # nothing to encode stages nothing
    base = torch.arange(2 * 4000, dtype=torch.float32).reshape(2, 4000)
    view = base[..., :3000]
    encodings.encode([(view, 1000), (base, 1000), (base, 1000)])
    assert entered == [vae]
    assert len(encodings) == 3 and len(vae.calls) == 3
    changed = base.clone()
    changed[1, 5] += 1
    for waveform, rate in (
        (view, 2000),
        (view.double(), 1000),
        (view.contiguous(), 1000),  # same values, other strides
        (changed[..., :3000], 1000),
    ):
        assert encodings.take(waveform, rate) is None
    # Equal dtype, shape, strides and values: another tensor still matches.
    rows, length = encodings.take(base.clone()[..., :3000], 1000)
    assert length == 120
    assert torch.equal(rows, FakeAudioVAE().encode_waveform(view, 1000)[0])
    assert encodings.take(base, 1000) is not None
    assert encodings.take(base, 1000) is not None
    assert encodings.take(base, 1000) is None
    assert len(encodings) == 0


def test_encode_media_reuses_early_rows_without_staging_the_wvae():
    media = _media()
    serial = processing.encode_media(media, video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(), emit_conditioning=True)
    vae = FakeAudioVAE()
    scope, entered = _recording_scope()
    encodings = processing.ReferenceAudioEncodings(vae, component_scope=scope)
    encodings.encode(_audio_inputs(media))
    assert entered == [vae] and len(vae.calls) == 2
    early = processing.encode_media(
        media,
        video_vae=FakeVideoVAE(),
        audio_vae=vae,
        emit_conditioning=True,
        component_scope=scope,
        reference_audio=encodings,
    )
    assert len(vae.calls) == 2
    assert entered.count(vae) == 1
    assert len(encodings) == 0
    assert serial.audio_condition_lengths == (120, 100)
    _assert_same_conditioning(early, serial)


def test_encode_media_encodes_what_the_early_rows_miss():
    media = _media()
    serial = processing.encode_media(media, video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(), emit_conditioning=True)
    vae = FakeAudioVAE()
    scope, entered = _recording_scope()
    encodings = processing.ReferenceAudioEncodings(vae)
    encodings.encode(_audio_inputs(media)[:1])
    early = processing.encode_media(
        media,
        video_vae=FakeVideoVAE(),
        audio_vae=vae,
        emit_conditioning=True,
        component_scope=scope,
        reference_audio=encodings,
    )
    assert len(vae.calls) == 2
    assert entered.count(vae) == 1
    _assert_same_conditioning(early, serial)


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required for reference-media tests")
    encoders = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=False, timeout=30
    ).stdout
    if "libx264" not in encoders or "libmp3lame" not in encoders:
        pytest.skip("ffmpeg with libx264 and libmp3lame is required for reference-media tests")
    root = tmp_path_factory.mktemp("minimax_h3_audio_encode_early")
    base = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi", "-i"]
    paths = {
        "video": str(root / "reference.mp4"),
        "silent": str(root / "silent.mp4"),
        "mp3": str(root / "reference.mp3"),
    }
    subprocess.run(
        [*base, "testsrc2=s=384x256:r=30", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100"]
        + ["-t", "6", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", paths["video"]],
        check=True,
        timeout=60,
    )
    subprocess.run(
        [*base, "testsrc2=s=384x256:r=24", "-t", "5", "-c:v", "libx264", "-preset", "ultrafast", paths["silent"]],
        check=True,
        timeout=60,
    )
    subprocess.run(
        [*base, "sine=frequency=330:sample_rate=44100", "-t", "3", "-c:a", "libmp3lame", paths["mp3"]],
        check=True,
        timeout=60,
    )
    return paths


def _prepare(sources, **kwargs):
    return processing.prepare_encoder_inputs(
        {"prompt": "reference", "multi_modal_data": {"video": [sources["video"]], "audio": sources["mp3"]}},
        OmniDiffusionSamplingParams(height=256, width=384, extra_args={"task": "ref2va", "duration": 4.0}),
        **kwargs,
    )


def _sha(value) -> str:
    array = value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _digest(prepared) -> dict:
    media = prepared.media
    return {
        "videos": [_sha(video) for video in media.videos],
        "video_audios": [None if item is None else (item[1], _sha(item[0])) for item in media.video_audios],
        "audios": [(rate, _sha(waveform)) for waveform, rate in media.audios],
        "qwen_videos": [(_sha(frames), meta) for frames, meta in prepared.qwen_videos],
        "condition_labels": prepared.condition_labels,
    }


def _encode(prepared, audio_vae, **kwargs):
    return processing.encode_media(
        prepared.media, video_vae=FakeVideoVAE(), audio_vae=audio_vae, emit_conditioning=True, **kwargs
    )


def test_prepare_encodes_reference_audio_while_the_frames_decode(monkeypatch, sources):
    monkeypatch.setenv(WORKERS_ENV, "3")
    events = []
    result = reference_video.ReferenceVideoFrames.result
    monkeypatch.setattr(
        reference_video.ReferenceVideoFrames, "result", lambda self: events.append("frames") or result(self)
    )
    vae = FakeAudioVAE(events=events)
    encodings = processing.ReferenceAudioEncodings(vae)
    threads = torch.get_num_threads()
    set_threads = []
    monkeypatch.setattr(processing.torch, "set_num_threads", set_threads.append)
    prepared = _prepare(sources, reference_audio=encodings)
    # Both references are encoded before this thread first waits for frames,
    # on this thread (the WVAE's determinism flags are process-global) and at
    # its own intra-op thread count, as the codec phase would encode them.
    assert events == ["encode", "encode", "frames"]
    assert len(encodings) == 2
    assert vae.idents == [threading.get_ident()] * 2
    assert vae.threads == [threads, threads] and set_threads == []
    serial = _prepare(sources)
    assert _digest(prepared) == _digest(serial)
    early = _encode(prepared, vae, reference_audio=encodings)
    assert len(vae.calls) == 2 and len(encodings) == 0
    _assert_same_conditioning(early, _encode(serial, FakeAudioVAE()))


def test_reference_audio_is_not_encoded_early_without_frame_jobs(sources):
    # A prepared-video descriptor decodes the frames on this thread: nothing to overlap.
    descriptor = {
        "original_path": sources["video"],
        "prepared_path": sources["video"],
        "input_has_audio": True,
        "audio_duration_seconds": 5.0,
    }
    vae = FakeAudioVAE()
    encodings = processing.ReferenceAudioEncodings(vae)
    _prepare(sources, prepared_reference_videos=[descriptor], reference_audio=encodings)
    assert vae.calls == [] and len(encodings) == 0


def test_silent_reference_video_stages_nothing(monkeypatch, sources):
    monkeypatch.setenv(WORKERS_ENV, "2")
    vae = FakeAudioVAE()
    scope, entered = _recording_scope()
    encodings = processing.ReferenceAudioEncodings(vae, component_scope=scope)
    processing.prepare_encoder_inputs(
        {"prompt": "reference", "multi_modal_data": {"video": [sources["silent"]]}},
        OmniDiffusionSamplingParams(height=256, width=384, extra_args={"task": "ref2va", "duration": 4.0}),
        reference_audio=encodings,
    )
    assert vae.calls == [] and entered == [] and len(encodings) == 0


def test_early_encode_failure_is_left_to_the_codec_phase(monkeypatch, sources):
    monkeypatch.setenv(WORKERS_ENV, "2")
    vae = FakeAudioVAE(fail=1)
    encodings = processing.ReferenceAudioEncodings(vae)
    prepared = _prepare(sources, reference_audio=encodings)
    assert len(vae.calls) == 1 and len(encodings) == 0
    early = _encode(prepared, vae, reference_audio=encodings)
    assert len(vae.calls) == 3
    _assert_same_conditioning(early, _encode(prepared, FakeAudioVAE()))
    with pytest.raises(RuntimeError, match="audio WVAE failed"):
        _encode(prepared, FakeAudioVAE(fail=1), reference_audio=processing.ReferenceAudioEncodings(FakeAudioVAE()))


@pytest.mark.parametrize(("raw", "encodes_early"), [(None, True), ("1", True), ("0", False)])
def test_pipeline_hands_early_rows_to_the_codec_phase(monkeypatch, raw, encodes_early):
    from PIL import Image

    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module
    from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderMediaConditioning

    if raw is None:
        monkeypatch.delenv(processing.MINIMAX_H3_AUDIO_ENCODE_EARLY_ENV, raising=False)
    else:
        monkeypatch.setenv(processing.MINIMAX_H3_AUDIO_ENCODE_EARLY_ENV, raw)
    pipeline = object.__new__(module.MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.load_text_encoder = True
    pipeline.load_vae_encoder = True
    pipeline.device = torch.device("cpu")
    pipeline.od_config = None
    pipeline._active_turbo_spec = lambda sampling: None
    pipeline._has_active_native_lora = lambda sampling: False
    pipeline._resolve_task = lambda *args, **kwargs: "ref2va"

    def encode_prompt(prepared):
        return torch.zeros(3, 5120, dtype=torch.bfloat16), torch.zeros(3, dtype=torch.long)

    pipeline.encode_prompt = encode_prompt
    pipeline.audio_vae = FakeAudioVAE()
    handed = []
    prepare = module.prepare_encoder_inputs

    def prepare_and_encode(*args, reference_audio=None, **kwargs):
        handed.append(reference_audio)
        if reference_audio is not None:
            reference_audio.encode([(torch.zeros(2, 32000), 16000)])
        return prepare(*args, **kwargs)

    def encode_media(media, **kwargs):
        handed.append(kwargs)
        return MiniMaxH3EncoderMediaConditioning(
            task=media.task,
            height=media.height,
            width=media.width,
            num_frames=media.num_frames,
            latent_t=media.latent_t,
            audio_t=media.audio_t,
        )

    monkeypatch.setattr(module, "prepare_encoder_inputs", prepare_and_encode)
    pipeline._encode_local_media = encode_media
    sampling = OmniDiffusionSamplingParams(width=64, height=64, fps=24, extra_args={"task": "ref2va", "duration": 4.4})
    request = {"prompt": "reference", "multi_modal_data": {"image": Image.new("RGB", (256, 256))}}
    pipeline._prepare_local_conditioning(request, sampling)
    reference_audio, media_kwargs = handed
    if encodes_early:
        assert isinstance(reference_audio, processing.ReferenceAudioEncodings)
        assert reference_audio._audio_vae is pipeline.audio_vae
        assert media_kwargs == {"reference_audio": reference_audio}
    else:
        assert reference_audio is None
        assert media_kwargs == {}


def test_pipeline_encodes_early_only_with_an_encoding_wvae(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    monkeypatch.delenv(processing.MINIMAX_H3_AUDIO_ENCODE_EARLY_ENV, raising=False)
    pipeline = object.__new__(module.MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    assert pipeline._early_reference_audio() is None
    pipeline.audio_vae = SimpleNamespace(decode_only=True)
    assert pipeline._early_reference_audio() is None
    pipeline.audio_vae = SimpleNamespace(decode_only=False)
    assert isinstance(pipeline._early_reference_audio(), processing.ReferenceAudioEncodings)

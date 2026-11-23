# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Duration limits apply before encoding and to transported encoder payloads."""

import os
import tempfile
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.minimax_h3 import reference_video
from vllm_omni.model_executor.models.minimax_h3.encoder_processing import (
    prepare_encoder_inputs,
    resolve_minimax_h3_shape,
)
from vllm_omni.model_executor.models.minimax_h3.long_video import (
    MINIMAX_H3_MAX_FULL_SECONDS_ENV,
    max_full_seconds,
    max_output_seconds,
    max_reference_seconds,
    max_request_reference_seconds,
    validate_encoded_frame_limit,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("mode,limit", [("full", 30), ("continuation", 300), (None, 300)])
@pytest.mark.parametrize("input_kind", ["duration", "duration_seconds", "target", "num_frames"])
def test_long_video_limits_cover_all_duration_inputs(mode, limit, input_kind):
    extra: dict[str, Any] = {"long_video": True}
    if mode is not None:
        extra["long_video_mode"] = mode
    sampling = OmniDiffusionSamplingParams(width=64, height=64, fps=24, extra_args=extra)

    def set_duration(seconds):
        if input_kind == "target":
            extra["target"] = {"duration_seconds": seconds}
        elif input_kind == "num_frames":
            sampling.num_frames = seconds * 24
        else:
            extra[input_kind] = seconds

    set_duration(limit)
    _, _, frames, _, _ = resolve_minimax_h3_shape("ref2va", sampling, None)
    assert frames >= limit * 24
    validate_encoded_frame_limit(extra, "ref2va", frames)
    with pytest.raises(OmniClientError, match="request limit"):
        validate_encoded_frame_limit(extra, "ref2va", frames + 17)
    set_duration(limit + 1)
    with pytest.raises(OmniClientError, match="duration"):
        resolve_minimax_h3_shape("ref2va", sampling, None)


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
def test_num_frames_cap_matches_aligned_max_duration(task):
    image = Image.new("RGB", (64, 64)) if task == "fl2va" else None
    sampling = OmniDiffusionSamplingParams(
        width=64, height=64, fps=24, num_frames=362, extra_args={"aspect_ratio": "1:1"}
    )
    assert resolve_minimax_h3_shape(task, sampling, image)[2] == 362
    sampling.extra_args = {"aspect_ratio": "1:1", "duration": 15}
    assert resolve_minimax_h3_shape(task, sampling, image)[2] == 362
    sampling.extra_args = {"aspect_ratio": "1:1"}
    sampling.num_frames = 363
    with pytest.raises(OmniClientError, match="362 frames"):
        resolve_minimax_h3_shape(task, sampling, image)


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
def test_full_mode_rejects_300_seconds_and_accepts_comparison(task):
    sampling = OmniDiffusionSamplingParams(
        width=64,
        height=64,
        fps=24,
        extra_args={"long_video": True, "long_video_mode": "full", "duration": 300, "aspect_ratio": "1:1"},
    )
    image = Image.new("RGB", (64, 64)) if task == "fl2va" else None
    with pytest.raises(OmniClientError, match="30"):
        resolve_minimax_h3_shape(task, sampling, image)
    sampling.extra_args["duration"] = 20
    assert resolve_minimax_h3_shape(task, sampling, image)[2] == 481


@pytest.mark.parametrize("mode", ["full", "continuation"])
def test_mode_alone_does_not_bypass_long_video_opt_in(mode):
    sampling = OmniDiffusionSamplingParams(
        width=64,
        height=64,
        extra_args={"long_video_mode": mode, "duration": 20},
    )
    with pytest.raises(OmniClientError, match="15"):
        resolve_minimax_h3_shape("ref2va", sampling, None)


@pytest.mark.parametrize("kind", ["video", "audio"])
def test_continuation_edit_rejected_before_loading_source(kind):
    sampling = OmniDiffusionSamplingParams(
        width=64,
        height=64,
        extra_args={"long_video": True, "task": "ref2va", "duration": 20},
    )
    with pytest.raises(OmniClientError, match="continuation does not support latent-mask editing"):
        prepare_encoder_inputs(
            {
                "prompt": "test",
                "multi_modal_data": {
                    "image": Image.new("RGB", (64, 64)),
                    f"source_{kind}": "must-not-be-loaded",
                    f"{kind}_noise_mask": 0.5,
                },
            },
            sampling,
        )


def test_locked_audio_rejects_audio_edit_before_loading_source():
    sampling = OmniDiffusionSamplingParams(
        width=64,
        height=64,
        extra_args={"task": "t2va", "aspect_ratio": "1:1", "audio_mode": "lock_source"},
    )
    with pytest.raises(OmniClientError, match="cannot be combined"):
        prepare_encoder_inputs(
            {
                "prompt": "test",
                "multi_modal_data": {
                    "audio": (torch.zeros(8000), 8000),
                    "source_audio": "must-not-be-loaded",
                    "audio_noise_mask": 0.5,
                },
            },
            sampling,
        )


@pytest.mark.parametrize("kind", ["video", "audio"])
def test_external_encoder_cannot_bypass_continuation_edit_validation(kind):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(step_execution=False)
    conditioning = SimpleNamespace(
        task="ref2va",
        num_frames=481,
        video_edit_clean_rows=torch.zeros(1) if kind == "video" else None,
        audio_edit_clean_rows=torch.zeros(1) if kind == "audio" else None,
    )
    sampling = OmniDiffusionSamplingParams(extra_args={"long_video": True})
    with pytest.raises(OmniClientError, match="continuation does not support latent-mask editing"):
        pipeline._prepare_encoder_conditioning_inputs(conditioning, sampling)


def test_external_encoder_cannot_bypass_full_duration_limit():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(step_execution=False)
    sampling = OmniDiffusionSamplingParams(extra_args={"long_video": True, "long_video_mode": "full"})
    with pytest.raises(OmniClientError, match="30-second request limit"):
        pipeline._prepare_encoder_conditioning_inputs(SimpleNamespace(task="ref2va", num_frames=7213), sampling)


_FULL = {"long_video": True, "long_video_mode": "full"}


@pytest.mark.parametrize(
    ("extra", "task", "num_frames", "seconds"),
    [
        ({}, "ref2va", 362, 15),
        ({"long_video_mode": "full"}, "ref2va", 736, 15),
        ({"long_video": True}, "ref2va", 736, 15),
        ({"long_video": True, "long_video_mode": "continuation"}, "ref2va", 736, 15),
        (_FULL, "ref2va", 362, 15),
        (_FULL, "ref2va", 379, 30),
        (_FULL, "ref2va", 736, 30),
        ({"long_video": True}, "t2va", 736, 30),
    ],
)
def test_reference_limit_opens_only_for_long_full_outputs(extra, task, num_frames, seconds):
    assert max_request_reference_seconds(extra, task, num_frames) == seconds
    assert max_reference_seconds(num_frames) >= seconds


def test_encoded_soundtrack_limit_is_unchanged_on_the_short_grid():
    assert [max_reference_seconds(frames) for frames in (124, 362, 379, 736)] == [15, 15, 30, 30]


def _prepare_references(monkeypatch, tmp_path, durations, **kwargs):
    meta = {
        "width": 1280,
        "height": 720,
        "fps": 24.0,
        "format_names": ("mp4",),
        "video_codec": "h264",
        "audio_codecs": ("aac",),
        "file_size": 1024,
    }

    def probe(path):
        duration = durations[os.path.basename(path)]
        return {**meta, "duration": duration, "frame_count": int(duration * 24)}

    monkeypatch.setattr(reference_video, "_probe_video", probe)
    monkeypatch.setattr(reference_video, "_transcode_reference_video", lambda source, **_: source)
    return reference_video.prepare_reference_videos(
        [str(tmp_path / name) for name in durations],
        target_frame_count=736,
        workdir=tempfile.mkdtemp(dir=tmp_path),
        **kwargs,
    )


def test_reference_videos_accept_the_long_full_budget_only_when_asked(monkeypatch, tmp_path):
    with pytest.raises(OmniClientError, match=r"\[2, 15\] seconds"):
        _prepare_references(monkeypatch, tmp_path, {"a.mp4": 30.0})
    with pytest.raises(OmniClientError, match="at most 15 seconds in total"):
        _prepare_references(monkeypatch, tmp_path, {"a.mp4": 10.0, "b.mp4": 10.0})
    (item,) = _prepare_references(monkeypatch, tmp_path, {"a.mp4": 30.0}, max_total_seconds=30.0)
    assert item["duration_seconds"] == item["audio_duration_seconds"] == 30.0
    assert len(_prepare_references(monkeypatch, tmp_path, {"a.mp4": 15.0, "b.mp4": 15.0}, max_total_seconds=30.0)) == 2
    with pytest.raises(OmniClientError, match=r"\[2, 30\] seconds"):
        _prepare_references(monkeypatch, tmp_path, {"a.mp4": 31.0}, max_total_seconds=30.0)
    with pytest.raises(OmniClientError, match="at most 30 seconds in total"):
        _prepare_references(monkeypatch, tmp_path, {"a.mp4": 20.0, "b.mp4": 11.0}, max_total_seconds=30.0)


def test_soundtrack_waveforms_follow_the_reference_limit():
    rate = 16000
    thirty = [(torch.zeros(1, 30 * rate), rate)]
    with pytest.raises(OmniClientError, match=r"\[2, 15\] seconds"):
        reference_video.validate_reference_audio_waveforms(thirty)
    reference_video.validate_reference_audio_waveforms(thirty, max_seconds=30.0)
    with pytest.raises(OmniClientError, match="at most 30 seconds in total"):
        reference_video.validate_reference_audio_waveforms([(torch.zeros(1, 16 * rate), rate)] * 2, max_seconds=30.0)


def test_full_mode_cap_reads_its_environment_override(monkeypatch):
    sampling = OmniDiffusionSamplingParams(width=64, height=64, fps=24, extra_args={**_FULL, "duration": 45})
    with pytest.raises(OmniClientError, match="30"):
        resolve_minimax_h3_shape("ref2va", sampling, None)
    monkeypatch.setenv(MINIMAX_H3_MAX_FULL_SECONDS_ENV, "45")
    assert max_output_seconds(_FULL, "ref2va") == 45
    assert max_output_seconds({}, "ref2va") == 15
    frames = resolve_minimax_h3_shape("ref2va", sampling, None)[2]
    assert frames >= 45 * 24
    validate_encoded_frame_limit(_FULL, "ref2va", frames)
    assert max_request_reference_seconds(_FULL, "ref2va", frames) == 45
    assert max_reference_seconds(362) == 15
    for raw in ("10", "61", "nan", "inf", "abc"):
        monkeypatch.setenv(MINIMAX_H3_MAX_FULL_SECONDS_ENV, raw)
        with pytest.raises(ValueError, match=MINIMAX_H3_MAX_FULL_SECONDS_ENV):
            max_full_seconds()
    monkeypatch.delenv(MINIMAX_H3_MAX_FULL_SECONDS_ENV)
    assert max_full_seconds() == 30

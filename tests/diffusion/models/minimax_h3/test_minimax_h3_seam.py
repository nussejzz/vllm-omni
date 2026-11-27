# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3 import continuation as continuation_module
from vllm_omni.diffusion.models.minimax_h3.continuation import (
    ContinuationWindow,
    SeamOptions,
    _audio_t,
    _video_t,
    diffuse_continuation,
    plan_continuation_windows,
    resolve_seam,
    seam_window_args,
    window_args,
)
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import minimax_h3_denoise_loop
from vllm_omni.diffusion.models.minimax_h3.latent_cache import CachedClip, plan_latent_continuation
from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence_ref2va_blocks
from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
    minimax_h3_pack_audio_latent,
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from vllm_omni.diffusion.models.minimax_h3.pika import PIKA_CORRELATED_SEED_OFFSETS, pika_sigmas
from vllm_omni.diffusion.models.minimax_h3.sampling import MiniMaxH3Model
from vllm_omni.diffusion.models.minimax_h3.scheduling_minimax_h3_euler_ancestral import minimax_h3_rf_v_to_x0
from vllm_omni.diffusion.sampler import RENOISE_SAMPLER, renoise_step
from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.models.minimax_h3.continuation_request import ContinuationRequest

from .test_minimax_h3_latent_cache import LATENT_H, LATENT_W, _clip, _conditioning, _pipeline, _run, _sampler
from .test_minimax_h3_step_execution import _make_branch, _SegmentMeanModel, _step_pipeline
from .test_pika_h3 import _noise, _RecordingModel, _sampling, _spec

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_WINDOWS = (277, 22)
_CHAIN = {"continue_from": "clip-1"}


def _kwargs(total: int, **extra):
    return dict(
        num_frames=total,
        latent_t=_video_t(total),
        latent_h=LATENT_H,
        latent_w=LATENT_W,
        audio_t=_audio_t(total),
        seed=500000,
        **extra,
    )


def _upstream_plan(total, window, overlap, *, first_overlap=0, origin=0):
    end = min(total, window)
    windows = [ContinuationWindow(origin, origin + end, first_overlap)]
    while end < total:
        start = end - overlap
        end = min(total, start + window)
        windows.append(ContinuationWindow(origin + start, origin + end, overlap))
    return windows


@pytest.mark.parametrize(
    "seam,chain,continuation,renoise,step,match",
    [
        ({}, False, None, True, False, "requires long_video_mode=continuation or continue_from"),
        ([], True, None, True, False, "must be an object"),
        ({"pins": "all"}, False, _WINDOWS, True, False, "unknown keys: pins"),
        ({}, True, None, True, True, "requires request execution"),
        ({"pin": "half"}, False, _WINDOWS, True, False, "pin must be one of"),
        ({"pin": "all"}, False, _WINDOWS, False, False, "re-noising sampler"),
        ({"pin": "chunk"}, True, _WINDOWS, True, False, "pin=chunk requires in-request continuation"),
        ({"pin": "only"}, True, None, True, False, "pin=only requires in-request continuation"),
        ({"pin": "chunk"}, False, (277, 5), True, False, "continuation_overlap_frames >= 22"),
        ({"seed_stride": 10006}, False, _WINDOWS, True, False, "seed_stride must be 0 or an integer"),
        ({"seed_stride": True}, False, _WINDOWS, True, False, "seed_stride must be 0 or an integer"),
        ({"seed_stride": 2**31}, False, _WINDOWS, True, False, "seed_stride must be 0 or an integer"),
        ({"seed_stride": -10007}, False, _WINDOWS, True, False, "seed_stride must be 0 or an integer"),
        ({"seed_stride": 10007}, True, None, True, False, "applies to in-request windows"),
        ({"clock": "local"}, False, _WINDOWS, True, False, "clock must be global or window"),
        ({"clock": "window"}, False, _WINDOWS, True, False, "clock=window requires h3_seam.seed_stride"),
        ({"lookahead": 16}, False, _WINDOWS, True, False, "lookahead must be 0, 17 or 34"),
        ({"lookahead": 17.0}, False, _WINDOWS, True, False, "lookahead must be 0, 17 or 34"),
        ({"lookahead": 34}, False, (107, 73), True, False, "overlap plus h3_seam.lookahead by 17 frames"),
        ({"return_overlap": 1}, True, None, True, False, "return_overlap must be a boolean"),
        ({"return_overlap": True}, False, _WINDOWS, True, False, "return_overlap requires continue_from"),
        ({"diag": "yes"}, False, _WINDOWS, True, False, "diag must be a boolean"),
    ],
)
def test_invalid_seam_options(seam, chain, continuation, renoise, step, match):
    extra = {"h3_seam": seam, **(_CHAIN if chain else {})}
    with pytest.raises(OmniClientError, match=match):
        resolve_seam(extra, continuation=continuation, renoise=renoise, step_execution=step)


def test_seam_options_resolve_on_both_paths():
    for continuation in (None, _WINDOWS):
        assert resolve_seam({}, continuation=continuation, renoise=True, step_execution=True) is None
        assert resolve_seam(_CHAIN, continuation=continuation, renoise=True, step_execution=False) is None
    assert resolve_seam({"h3_seam": {}}, continuation=_WINDOWS, renoise=False, step_execution=False) == SeamOptions()
    chain = {**_CHAIN, "h3_seam": {"return_overlap": True, "clock": "window", "lookahead": 17, "pin": "all"}}
    assert resolve_seam(chain, continuation=None, renoise=True, step_execution=False) == SeamOptions(
        pin="all", clock="window", lookahead=17, return_overlap=True
    )
    both = {**_CHAIN, "h3_seam": {"seed_stride": 10007, "clock": "window", "lookahead": 34}}
    assert resolve_seam(both, continuation=_WINDOWS, renoise=False, step_execution=False) == SeamOptions(
        seed_stride=10007, clock="window", lookahead=34
    )
    with pytest.raises(OmniClientError, match="continue_overlap_frames"):
        resolve_seam(
            {**_CHAIN, "continue_overlap_frames": 20, "h3_seam": {}},
            continuation=None,
            renoise=True,
            step_execution=False,
        )


@pytest.mark.parametrize(
    "overlap,lookahead,accepted",
    [(243, 17, True), (260, 0, True), (260, 17, False), (260, 34, False), (243, 34, False), (277, 0, False)],
)
def test_chain_overlap_and_lookahead_leave_new_frames_in_window_zero(overlap, lookahead, accepted):
    extra = {**_CHAIN, "continue_overlap_frames": overlap, "h3_seam": {"lookahead": lookahead}}
    seam = SeamOptions(lookahead=lookahead)
    assert resolve_seam(extra, continuation=None, renoise=True, step_execution=False) == seam
    if accepted:
        assert resolve_seam(extra, continuation=_WINDOWS, renoise=True, step_execution=False) == seam
    else:
        with pytest.raises(OmniClientError, match="exceed continue_overlap_frames plus h3_seam.lookahead"):
            resolve_seam(extra, continuation=_WINDOWS, renoise=True, step_execution=False)


@pytest.mark.parametrize("total", [124, 277, 736, 1450, 2895])
@pytest.mark.parametrize("window,overlap", [(107, 22), (277, 22), (243, 39), (345, 56)])
def test_lookahead_zero_plan_is_upstream(total, window, overlap):
    assert plan_continuation_windows(total, window, overlap) == _upstream_plan(total, window, overlap)
    assert plan_continuation_windows(total, window, overlap, lookahead=0) == _upstream_plan(total, window, overlap)
    assert plan_continuation_windows(
        total, window, overlap, first_overlap=22, origin=255, lookahead=0
    ) == _upstream_plan(total, window, overlap, first_overlap=22, origin=255)


@pytest.mark.parametrize("lookahead", [17, 34])
@pytest.mark.parametrize("total", [532, 736, 1807, 2895])
def test_lookahead_plan_keeps_the_grid(lookahead, total):
    windows = plan_continuation_windows(total, 277, 22, lookahead=lookahead)
    assert windows[0] == ContinuationWindow(0, 277, 0) and windows[-1].end == total
    for previous, window in zip(windows, windows[1:]):
        assert window.start == previous.end - 22 - lookahead and window.overlap == 22
    for window in windows:
        assert window.start % 17 == 0 and (window.end - window.start) % 17 == 5
    if total == 1807:
        assert len(windows) == 8 and len(plan_continuation_windows(total, 277, 22)) == 7
    for bad in (16, 255):
        with pytest.raises(ValueError, match="lookahead"):
            plan_continuation_windows(total, 277, 22, lookahead=bad)
    with pytest.raises(ValueError, match="lookahead"):
        plan_continuation_windows(total, 107, 73, lookahead=34)


_CONTINUATIONS = [
    {"long_video_mode": "continuation"},
    {"continuation_prompts": ["a", "b"]},
    {"continue_from": "clip-123"},
]


@pytest.mark.parametrize("extra", _CONTINUATIONS)
def test_pika_ref2va_unlocks_continuation_only_with_h3_seam(extra):
    with pytest.raises(ValueError):
        _spec().check_request(_sampling(extra_args=dict(extra)), "ref2va")
    with pytest.raises(ValueError):
        _spec().check_request(_sampling(extra_args={**extra, "h3_seam": None}), "ref2va")
    _spec().check_request(_sampling(extra_args={**extra, "h3_seam": {}}), "ref2va")


@pytest.mark.parametrize("extra", _CONTINUATIONS)
def test_pika_fl2va_rejects_continuation_exactly_as_before(extra):
    messages = []
    for seam in (None, {}, {"pin": "all"}):
        request = dict(extra) if seam is None else {**extra, "h3_seam": seam}
        with pytest.raises(ValueError) as info:
            _spec("fl2va").check_request(_sampling(extra_args=request), "t2va")
        messages.append(str(info.value))
    assert len(set(messages)) == 1


@pytest.mark.parametrize("seam", [{}, {"pin": "chunk"}, {"pins": 1}, {"seed_stride": 5}, []])
@pytest.mark.parametrize("task,extra", [("t2va", {"continue_from": "a"}), ("fl2va", {})])
def test_pika_fl2va_pipeline_rejects_exactly_as_before(task, extra, seam):
    pipeline, calls, _ = _pipeline()
    pipeline._pika_spec = _spec("fl2va")
    messages = []
    for request in (extra, {**extra, "h3_seam": seam}):
        with pytest.raises(OmniClientError) as info:
            _run(pipeline, _conditioning(task, 141), dict(request))
        messages.append(str(info.value))
    assert messages[0] == messages[1] and not calls


@pytest.mark.parametrize("continuation", [{"continue_from": "clip-123"}, {"long_video_mode": "continuation"}])
@pytest.mark.parametrize("extra", [{"audio_mode": "lock_source"}, {"sampler": "res_multistep"}, {"regenerate": True}])
def test_h3_seam_keeps_other_pika_rejections(continuation, extra):
    with pytest.raises(ValueError):
        _spec().check_request(_sampling(extra_args={**continuation, "h3_seam": {}, **extra}), "ref2va")


def _pika_inputs(**overrides):
    spec = _spec()
    kwargs = dict(
        task="ref2va",
        text_embeddings=torch.ones(3, 8),
        text_tags=torch.ones(3, dtype=torch.long),
        seed=42,
        latent_t=2,
        latent_h=4,
        latent_w=6,
        audio_t=3,
        num_frames=22,
        num_steps=6,
        video_shift=12.0,
        audio_shift=3.0,
        base_schedule=spec.base_schedule,
        visual_condition=torch.linspace(-1.001, 1.001, 6 * 96).reshape(6, 96),
        visual_condition_shape=(1, 4, 6),
        audio_condition=None,
        ref_audio_t=None,
        ref_blocks=[{"kind": "image", "latent_h": 4, "latent_w": 6}],
        pika=spec,
    )
    return _step_pipeline(_SegmentMeanModel())._build_denoise_inputs(**{**kwargs, **overrides})


def test_pika_continuation_geometry_needs_h3_seam():
    with pytest.raises(OmniClientError, match="timeline guides / continuation"):
        _pika_inputs(temporal_offset=425.0)
    plain = _pika_inputs()
    shifted = _pika_inputs(temporal_offset=425.0, seam=SeamOptions())
    for key in ("video_rows", "audio_rows", "cond_anchor"):
        torch.testing.assert_close(shifted[key], plain[key], rtol=0, atol=0)
    assert shifted["sampler"] == RENOISE_SAMPLER
    with pytest.raises(OmniClientError, match="driving audio"):
        _pika_inputs(temporal_offset=425.0, seam=SeamOptions(), locked_audio_rows=torch.zeros(6, 32))


@pytest.mark.parametrize("stride", [10007, 20011])
def test_seed_stride_windows(stride):
    calls: list[dict] = []
    video, audio = diffuse_continuation(
        _sampler(calls), _kwargs(736, seam=SeamOptions(seed_stride=stride)), window_frames=277, overlap_frames=22
    )
    assert [call["seed"] for call in calls] == [500000, 500000 + stride, 500000 + 2 * stride]
    assert [call["temporal_offset"] for call in calls] == pytest.approx([start * 40 / 24 for start in (0, 255, 510)])
    assert video.shape[2] == _video_t(736) and audio.shape[-1] == _audio_t(736)


def _target_positions(call):
    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=3,
        latent_t=call["latent_t"],
        latent_h=call["latent_h"],
        latent_w=call["latent_w"],
        audio_t=call["audio_t"],
        ref_blocks=call["ref_blocks"],
        temporal_offset=call["temporal_offset"],
        media_time_origin=call.get("media_time_origin"),
    )
    positions = packed["img_position_ids"]
    return (
        positions[packed["img_pos"][packed["update_mask"]]],
        positions[packed["audio_pos"][packed["audio_update_mask"]]],
    )


@pytest.mark.parametrize("clock", ["window", "global"])
def test_window_clock_windows_share_window_zero_grid(clock):
    patches = (LATENT_H // 2) * (LATENT_W // 2)
    kwargs = _kwargs(
        532,
        seam=SeamOptions(seed_stride=10007, clock=clock),
        ref_blocks=[{"kind": "image", "latent_h": LATENT_H, "latent_w": LATENT_W}],
        visual_condition=torch.zeros(patches, 96),
        visual_condition_shapes=[(1, LATENT_H, LATENT_W)],
    )
    calls: list[dict] = []
    diffuse_continuation(_sampler(calls), kwargs, window_frames=277, overlap_frames=22)
    assert len(calls) == 2 and calls[1]["ref_blocks"][-1]["kind"] == "latent_guide"
    first, second = (_target_positions(call) for call in calls)
    if clock == "window":
        assert all(call["temporal_offset"] == 0.0 and call["media_time_origin"] is None for call in calls)
        for a, b in zip(first, second, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    else:
        for a, b in zip(first, second, strict=True):
            torch.testing.assert_close(b[:, 0] - a[:, 0], torch.full_like(a[:, 0], 425.0))
            torch.testing.assert_close(b[:, 1:], a[:, 1:], rtol=0, atol=0)


@pytest.mark.parametrize("clock", ["window", "global"])
def test_window_clock_replaces_the_shared_prompt_origin(clock):
    texts = [(torch.full((3 + index, 8), float(index)), torch.ones(3 + index, dtype=torch.long)) for index in range(2)]
    calls: list[dict] = []
    diffuse_continuation(
        _sampler(calls),
        _kwargs(532, seam=SeamOptions(seed_stride=10007, clock=clock)),
        window_frames=277,
        overlap_frames=22,
        text_conditioning=texts,
    )
    assert [call["text_embeddings"].shape[0] for call in calls] == [3, 4]
    if clock == "window":
        assert [(call["temporal_offset"], call["media_time_origin"]) for call in calls] == [(0.0, None)] * 2
    else:
        assert [call["media_time_origin"] for call in calls] == [4, 4]
        assert [call["temporal_offset"] for call in calls] == pytest.approx([0.0, 425.0])


def test_renoise_with_pins_holds_the_tail_at_every_step():
    model = _RecordingModel()
    branch, video, audio = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=3)
    target_v = torch.linspace(-1, 1, 6 * 96).reshape(6, 96)
    target_a = torch.linspace(-2, 2, 2 * 32).reshape(2, 32)
    rows_v, rows_a = torch.arange(6), torch.tensor([0, 3])
    pins = ((rows_v, target_v, 1), (rows_a, target_a, 1))
    trace: list[dict] = []
    sv, sa = pika_sigmas(12.0), pika_sigmas(3.0)
    loop = dict(
        model=model,
        positive=branch,
        initial_video_rows=video,
        initial_audio_rows=audio,
        keyframe_cond_rows=None,
        sigmas_video=sv,
        sigmas_audio=sa,
        device=torch.device("cpu"),
    )
    out_video, out_audio = minimax_h3_denoise_loop(
        **loop, sampler=RENOISE_SAMPLER, noise_source=_noise(3), pins=pins, pin_trace=trace
    )
    assert len(model.calls) == 6
    reference = _noise(3)
    for i in range(1, 6):
        eps_v, eps_a = reference.draw(i - 1, (video, audio))
        expected_v = renoise_step(target_v, eps_v[rows_v], sigma_next=sv[i])
        expected_a = renoise_step(target_a, eps_a[rows_a], sigma_next=sa[i])
        torch.testing.assert_close(model.calls[i]["video_rows"][rows_v], expected_v, rtol=0, atol=0)
        torch.testing.assert_close(model.calls[i]["audio_rows"][rows_a], expected_a, rtol=0, atol=0)
    torch.testing.assert_close(out_video[rows_v], target_v, rtol=0, atol=0)
    torch.testing.assert_close(out_audio[rows_a], target_a, rtol=0, atol=0)
    assert len(trace) == 6
    assert [entry["sigma_video"] for entry in trace] == sv[:-1]
    assert [entry["sigma_audio"] for entry in trace] == sa[:-1]
    assert all(len(entry["video"]) == 1 and len(entry["audio"]) == 1 for entry in trace)
    with pytest.raises(ValueError, match="re-noising sampler"):
        minimax_h3_denoise_loop(**loop, pins=pins)


@pytest.mark.parametrize("pin", ["all", "chunk"])
def test_pins_are_the_tail_in_target_row_order(pin):
    window = ContinuationWindow(255, 532, 22)
    audio_t = window.audio_end - window.audio_start
    generator = torch.Generator().manual_seed(11)
    video_tail = torch.randn(1, 24, 7, LATENT_H, LATENT_W, generator=generator)
    audio_tail = torch.randn(2, 32, window.overlap_audio_t, generator=generator)
    args = seam_window_args(SeamOptions(pin=pin), {"seed": 0}, window, 1, video_tail, audio_tail)
    assert set(args) == {"pins"}
    (video_rows, video_values, groups), (audio_rows, audio_values, audio_groups) = args["pins"]
    count = 7 if pin == "all" else 5
    audio_count = window.overlap_audio_t if pin == "all" else _audio_t(window.start + 17) - window.audio_start
    assert groups == count and audio_groups == 1 and audio_rows.numel() == 2 * audio_count
    rows_v = torch.zeros(_video_t(277) * (LATENT_H // 2) * (LATENT_W // 2), 96)
    rows_v[video_rows] = video_values
    rows_a = torch.zeros(2 * audio_t, 32)
    rows_a[audio_rows] = audio_values
    latent = minimax_h3_unpatchify_video_tokens(
        rows_v, latent_shape=(_video_t(277), LATENT_H // 2, LATENT_W // 2, 24), patch_size=(1, 2, 2)
    )
    unpacked = minimax_h3_unpack_audio_tokens(rows_a, audio_t=2 * audio_t, audio_channel=2)
    assert torch.equal(latent[:, :, :count], video_tail[:, :, :count]) and torch.all(latent[:, :, count:] == 0)
    assert torch.equal(unpacked[..., :audio_count], audio_tail[..., :audio_count])
    assert torch.all(unpacked[..., audio_count:] == 0)


class _MaskedModel(_SegmentMeanModel):
    def __call__(self, **kwargs):
        outputs = super().__call__(**kwargs)
        self.calls[-1]["update_mask"] = kwargs["update_mask"].view(-1).bool().clone()
        return outputs


def _pipeline_windows(seam, total, *, window_frames=277, prefix=None, audio_t=None):
    spec = _spec()
    model = _MaskedModel()
    pipeline = _step_pipeline(model)
    pipeline.progress_bar = lambda total: contextlib.nullcontext(SimpleNamespace(update=lambda: None))
    generator = torch.Generator().manual_seed(1)
    kwargs = _kwargs(
        total,
        task="ref2va",
        text_embeddings=torch.randn(3, 8, generator=generator),
        text_tags=torch.ones(3, dtype=torch.long),
        num_steps=6,
        video_shift=12.0,
        audio_shift=3.0,
        base_schedule=spec.base_schedule,
        visual_condition=torch.randn((LATENT_H // 2) * (LATENT_W // 2), 96, generator=generator),
        visual_condition_shape=(1, LATENT_H, LATENT_W),
        visual_condition_shapes=[(1, LATENT_H, LATENT_W)],
        audio_condition=None,
        ref_audio_t=None,
        ref_blocks=[{"kind": "image", "latent_h": LATENT_H, "latent_w": LATENT_W}],
        pika=spec,
        seam=seam,
    )
    if audio_t is not None:
        kwargs["audio_t"] = audio_t
    outputs = []

    def diffuse(**args):
        outputs.append(pipeline.diffuse(**args))
        return outputs[-1]

    diffuse_continuation(diffuse, kwargs, window_frames=window_frames, overlap_frames=22, prefix=prefix)
    return outputs, model.calls


def _condition_rows(call):
    return call["video_rows"][~call["update_mask"]]


@pytest.mark.parametrize("pin", ["all", "chunk", "only"])
def test_pipeline_diffuse_holds_the_tail_next_to_condition_rows(pin):
    base_outputs, base_calls = _pipeline_windows(SeamOptions(), 532)
    outputs, calls = _pipeline_windows(SeamOptions(pin=pin), 532)
    assert len(calls) == len(base_calls) == 12
    assert all(torch.equal(a, b) for a, b in zip(outputs[0], base_outputs[0], strict=True))
    window = ContinuationWindow(255, 532, 22)
    count = 5 if pin == "chunk" else 7
    audio_count = _audio_t(272) - window.audio_start if pin == "chunk" else window.overlap_audio_t
    tail_v, tail_a = base_outputs[0][0][:, :, -7:], base_outputs[0][1][..., -window.overlap_audio_t :]
    video, audio = outputs[1]
    assert torch.equal(video[:, :, :count], tail_v[:, :, :count])
    assert torch.equal(audio[..., :audio_count], tail_a[..., :audio_count])
    assert not torch.equal(base_outputs[1][0][:, :, :count], tail_v[:, :, :count])
    if pin != "only":
        for call, base in zip(calls[6:], base_calls[6:], strict=True):
            assert torch.equal(_condition_rows(call), _condition_rows(base))


def test_pipeline_diffuse_holds_the_prefix_behind_the_timeline_guide():
    generator = torch.Generator().manual_seed(7)
    clip = CachedClip(
        video=torch.randn(1, 24, _video_t(277), LATENT_H, LATENT_W, generator=generator),
        audio=torch.randn(2, 32, _audio_t(277), generator=generator),
        start_frame=0,
        num_frames=277,
    )
    plan = plan_latent_continuation(
        clip, ContinuationRequest("a", 22), num_frames=141, latent_h=LATENT_H, latent_w=LATENT_W, lookahead=17
    )
    runs = [
        _pipeline_windows(
            SeamOptions(pin=pin, lookahead=17), 141, window_frames=141, prefix=plan.prefix, audio_t=plan.audio_t
        )
        for pin in ("none", "all")
    ]
    (base_outputs, base_calls), (outputs, calls) = runs
    assert len(calls) == len(base_calls) == 6
    video, audio = outputs[0]
    assert torch.equal(video[:, :, :7], plan.prefix.video)
    assert torch.equal(audio[..., : plan.prefix.audio.shape[-1]], plan.prefix.audio)
    assert not torch.equal(base_outputs[0][0][:, :, :7], plan.prefix.video)
    for call, base in zip(calls, base_calls, strict=True):
        assert torch.equal(_condition_rows(call), _condition_rows(base))


def test_grid_x0_is_bit_identical_without_pins():
    branch, video, audio = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=5)
    generator = torch.Generator().manual_seed(5)
    velocities = (torch.randn(video.shape, generator=generator), torch.randn(audio.shape, generator=generator))
    evaluation = SimpleNamespace(
        latents=(0.5 * video[branch.update_mask], 0.5 * audio[branch.audio_update_mask]),
        sigmas=(pika_sigmas(12.0)[2], pika_sigmas(3.0)[2]),
    )
    grid = MiniMaxH3Model(None, branch, (video.clone(), audio.clone()), grid_sigma_x0=True)
    prepared = grid.prepare(evaluation)
    expected = tuple(
        row[mask] + sigma * v.float()[mask]
        for row, mask, sigma, v in zip(prepared[0], grid.masks, prepared.sigmas, velocities, strict=True)
    )
    for actual, value in zip(grid.to_denoised(prepared, velocities), expected, strict=True):
        assert torch.equal(actual, value)
    base = MiniMaxH3Model(None, branch, (video.clone(), audio.clone()))
    prepared = base.prepare(evaluation)
    expected = tuple(
        minimax_h3_rf_v_to_x0(row[mask], v.float()[mask], row.new_tensor(t))
        for row, mask, t, v in zip(prepared[0], base.masks, prepared[2], velocities, strict=True)
    )
    for actual, value in zip(base.to_denoised(prepared, velocities), expected, strict=True):
        assert torch.equal(actual, value)
    pins = ((torch.arange(3), torch.full((3, 96), 7.0), 1), None)
    trace: list[dict] = []
    pinned = MiniMaxH3Model(
        None, branch, (video.clone(), audio.clone()), grid_sigma_x0=True, pins=pins, pin_trace=trace
    )
    prepared = pinned.prepare(evaluation)
    x0_video, x0_audio = pinned.to_denoised(prepared, velocities)
    unpinned_video, unpinned_audio = grid.to_denoised(grid.prepare(evaluation), velocities)
    assert torch.all(x0_video[:3] == 7.0) and torch.equal(x0_video[3:], unpinned_video[3:])
    assert torch.equal(x0_audio, unpinned_audio)
    error = float((unpinned_video[:3] - 7.0).norm() / torch.full((3, 96), 7.0).norm())
    assert len(trace) == 1 and error > 0 and trace[0]["video"] == pytest.approx([error], abs=1e-5)
    assert (trace[0]["sigma_video"], trace[0]["sigma_audio"]) == evaluation.sigmas and "audio" not in trace[0]
    with pytest.raises(ValueError, match="grid-sigma"):
        MiniMaxH3Model(None, branch, (video, audio), pins=pins)


def _seam_lines(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(continuation_module.logger, "info", lambda message, *args: lines.append(message % args))
    return lines


def _records(lines):
    prefix = "MiniMax H3 seam "
    return [json.loads(line.split(prefix, 1)[1]) for line in lines if line.startswith(prefix)]


def _ramp_sampler(calls):
    def sample(**kw):
        calls.append(kw)
        base = 1000.0 * len(calls) + 1.0
        video = base + torch.arange(kw["latent_t"], dtype=torch.float32).view(1, 1, -1, 1, 1)
        audio = base + torch.arange(kw["audio_t"], dtype=torch.float32).view(1, 1, -1)
        return (
            video.expand(1, 24, -1, kw["latent_h"], kw["latent_w"]).contiguous(),
            audio.expand(2, 32, -1).contiguous(),
        )

    return sample


def test_seam_log_line_is_json(monkeypatch):
    lines = _seam_lines(monkeypatch)
    calls: list[dict] = []
    diffuse_continuation(
        _ramp_sampler(calls), _kwargs(532, seam=SeamOptions(diag=True)), window_frames=277, overlap_frames=22
    )
    first, second = _records(lines)
    assert (first["index"], second["index"], first["windows"]) == (0, 1, 2)
    assert (second["start"], second["end"], second["overlap"], second["seed"]) == (255, 532, 22, 500000)
    assert second["temporal_offset"] == pytest.approx(425.0) and second["media_time_origin"] is None
    assert "adherence" not in first and "pin_trace" not in second
    tail_v = 1001.0 + torch.arange(_video_t(277) - 7, _video_t(277), dtype=torch.float32)
    new_v = 2001.0 + torch.arange(7, dtype=torch.float32)
    tail_a = 1001.0 + torch.arange(_audio_t(255), _audio_t(277), dtype=torch.float32)
    new_a = 2001.0 + torch.arange(tail_a.numel(), dtype=torch.float32)
    adherence = second["adherence"]
    assert adherence["video"] == pytest.approx(((new_v - tail_v) / tail_v).tolist(), abs=1e-5)
    assert adherence["video_motion"] == pytest.approx((1.0 / tail_v[:-1]).tolist(), abs=1e-5)
    assert adherence["audio"] == pytest.approx(float((new_a - tail_a).norm() / tail_a.norm()), abs=1e-5)
    motion_a = tail_a[1:] - tail_a[:-1]
    assert adherence["audio_motion"] == pytest.approx(float(motion_a.norm() / tail_a.norm()), abs=1e-5)
    assert all(record["wall_s"] >= 0 for record in (first, second))
    if not torch.accelerator.is_available():
        assert "peak_alloc_gib" not in first
    lines.clear()
    calls.clear()
    entry = {"sigma_video": 1.0, "sigma_audio": 1.0, "video": [0.5], "audio": [0.25]}
    inner = _sampler(calls)

    def traced(**kw):
        if "pin_trace" in kw:
            kw["pin_trace"].append(entry)
        return inner(**kw)

    diffuse_continuation(
        traced, _kwargs(532, seam=SeamOptions(pin="all", diag=True)), window_frames=277, overlap_frames=22
    )
    first, second = _records(lines)
    assert "pins" not in calls[0] and "pin_trace" not in calls[0] and "pin_trace" not in first
    assert second["pin_trace"] == [entry]


def test_no_seam_log_without_h3_seam(monkeypatch):
    lines = _seam_lines(monkeypatch)
    diffuse_continuation(_sampler([]), _kwargs(532), window_frames=277, overlap_frames=22)
    assert len(lines) == 2 and not _records(lines)


def test_return_overlap_returns_the_hidden_overlap():
    pipeline, calls, mp4 = _pipeline()
    _run(pipeline, _conditioning("t2va", 124), {"latent_cache_key": "a"})
    overlap_audio_t = _audio_t(124) - _audio_t(102)
    video, audio = _run(
        pipeline,
        _conditioning("ref2va", 141),
        {"latent_cache_key": "b", "continue_from": "a", "h3_seam": {"return_overlap": True}},
    )
    assert video.shape[1] == 141 and audio.shape[-1] == calls[1]["audio_t"] * 800
    assert pipeline._latent_clip_cache.get("b").num_frames == 141
    video, audio = _run(pipeline, _conditioning("ref2va", 141), {"continue_from": "a"})
    assert video.shape[1] == 119 and audio.shape[-1] == (calls[2]["audio_t"] - overlap_audio_t) * 800
    _run(
        pipeline,
        _conditioning("ref2va", 141),
        {"continue_from": "a", "preencode_mp4": True, "h3_seam": {"return_overlap": True}},
    )
    _run(pipeline, _conditioning("ref2va", 141), {"continue_from": "a", "preencode_mp4": True})
    assert mp4[0] is None and mp4[1]["frames"] == 22


@pytest.mark.parametrize("window_frames", [277, 243])
def test_chunk_pin_splices_at_local_latent_five(window_frames):
    total, start = 2 * window_frames - 22, window_frames - 22
    calls: list[dict] = []
    video, audio = diffuse_continuation(
        _sampler(calls), _kwargs(total, seam=SeamOptions(pin="chunk")), window_frames=window_frames, overlap_frames=22
    )
    assert "pins" not in calls[0] and calls[1]["ref_blocks"][-1]["kind"] == "latent_guide"
    video_pin, audio_pin = calls[1]["pins"]
    a5 = _audio_t(start + 17) - _audio_t(start)
    assert (a5 == _audio_t(17)) == (window_frames == 277)
    assert video_pin[2] == 5 and video_pin[0].numel() == 5 * (LATENT_H // 2) * (LATENT_W // 2)
    assert audio_pin[0].numel() == 2 * a5
    split_v = _video_t(window_frames) - 2
    assert video.shape[2] == _video_t(total)
    assert torch.all(video[:, :, :split_v] == 101) and torch.all(video[:, :, split_v:] == 102)
    split = _audio_t(start) + a5
    assert audio.shape[-1] == _audio_t(total)
    assert torch.all(audio[..., :split] == 101) and torch.all(audio[..., split:] == 102)


def test_pin_only_pins_the_tail_without_a_guide():
    calls: list[dict] = []
    video, audio = diffuse_continuation(
        _sampler(calls), _kwargs(532, seam=SeamOptions(pin="only")), window_frames=277, overlap_frames=22
    )
    reference: list[dict] = []
    diffuse_continuation(
        _sampler(reference), _kwargs(532, seam=SeamOptions(pin="all")), window_frames=277, overlap_frames=22
    )
    assert "pins" not in calls[0] and reference[1]["ref_blocks"][-1]["kind"] == "latent_guide"
    assert not {"ref_blocks", "visual_condition", "audio_condition", "guide_blocks"} & set(calls[1])
    for got, want in zip(calls[1]["pins"], reference[1]["pins"], strict=True):
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1]) and got[2] == want[2]
    assert torch.all(video[:, :, : _video_t(277)] == 101) and torch.all(video[:, :, _video_t(277) :] == 102)
    assert video.shape[2] == _video_t(532) and audio.shape[-1] == _audio_t(532)


@pytest.mark.parametrize("chained", [False, True])
@pytest.mark.parametrize("lookahead", [17, 34])
def test_lookahead_truncates_before_the_tail(lookahead, chained):
    prefix = None
    total = 736
    if chained:
        total = 702
        prefix = plan_latent_continuation(
            _clip(141), ContinuationRequest("a", 22), num_frames=total, latent_h=LATENT_H, latent_w=LATENT_W
        ).prefix
    origin = prefix.start_frame if chained else 0
    kwargs = _kwargs(total, seam=SeamOptions(lookahead=lookahead))
    kwargs["audio_t"] = _audio_t(origin + total) - _audio_t(origin)
    calls: list[dict] = []
    video, audio = diffuse_continuation(_sampler(calls), kwargs, window_frames=277, overlap_frames=22, prefix=prefix)
    windows = plan_continuation_windows(
        total, 277, 22, first_overlap=22 if chained else 0, origin=origin, lookahead=lookahead
    )
    assert len(calls) == len(windows) >= 3
    cuts = [window.start + window.overlap for window in windows[1:]]
    video_bounds = [7 if chained else 0, *(_video_t(cut - origin) for cut in cuts), kwargs["latent_t"]]
    audio_bounds = [
        prefix.audio.shape[-1] if chained else 0,
        *(_audio_t(cut) - _audio_t(origin) for cut in cuts),
        kwargs["audio_t"],
    ]
    if chained:
        assert torch.equal(video[:, :, :7], prefix.video) and torch.equal(audio[..., : audio_bounds[0]], prefix.audio)
    for index in range(len(windows)):
        assert torch.all(video[:, :, video_bounds[index] : video_bounds[index + 1]] == 101 + index)
        assert torch.all(audio[..., audio_bounds[index] : audio_bounds[index + 1]] == 101 + index)
    for index, call in enumerate(calls[1:]):
        assert torch.all(call["visual_condition"] == 101 + index)
        assert torch.all(call["audio_condition"] == 101 + index)
        assert call["temporal_offset"] == pytest.approx(windows[index + 1].start * 40 / 24)
    assert video.shape[2] == kwargs["latent_t"] and audio.shape[-1] == kwargs["audio_t"]


@pytest.mark.parametrize("lookahead", [0, 17, 34])
def test_continue_from_lookahead_moves_the_prefix_back(lookahead):
    clip = _clip(277, start=255)
    plan = plan_latent_continuation(
        clip, ContinuationRequest("a", 22), num_frames=141, latent_h=LATENT_H, latent_w=LATENT_W, lookahead=lookahead
    )
    end = clip.end_frame - lookahead
    start = end - 22
    assert (plan.prefix.start_frame, plan.prefix.overlap_frames) == (start, 22)
    video_end = _video_t(277 - lookahead)
    assert torch.equal(plan.prefix.video, clip.video[:, :, video_end - 7 : video_end])
    audio = clip.audio[..., _audio_t(start) - _audio_t(255) : _audio_t(end) - _audio_t(255)]
    assert torch.equal(plan.prefix.audio, audio)
    assert plan.audio_t == _audio_t(start + 141) - _audio_t(start)
    if lookahead == 0:
        assert torch.equal(plan.prefix.video, clip.video[:, :, -7:])
        assert torch.equal(plan.prefix.audio, clip.audio[..., -(_audio_t(532) - _audio_t(510)) :])
    with pytest.raises(OmniClientError, match="lookahead"):
        plan_latent_continuation(
            _clip(39), ContinuationRequest("a", 22), num_frames=141, latent_h=LATENT_H, latent_w=LATENT_W, lookahead=34
        )


def test_pika_pipeline_runs_continuation_only_with_h3_seam():
    pipeline, calls, _ = _pipeline()
    _run(pipeline, _conditioning("t2va", 124), {"latent_cache_key": "a"})
    pipeline._pika_spec = _spec()
    with pytest.raises(OmniClientError, match="continue_from"):
        _run(pipeline, _conditioning("ref2va", 141), {"continue_from": "a"})
    with pytest.raises(OmniClientError, match="continuation is not supported"):
        _run(pipeline, _conditioning("ref2va", 736), {"long_video": True, "long_video_mode": "continuation"})
    assert len(calls) == 1
    video, _ = _run(
        pipeline, _conditioning("ref2va", 141), {"continue_from": "a", "h3_seam": {"pin": "all", "clock": "window"}}
    )
    call = calls[1]
    assert call["seam"] == SeamOptions(pin="all", clock="window") and call["pika"] is pipeline._pika_spec
    assert call["temporal_offset"] == 0.0 and call["guide_blocks"][0]["kind"] == "video_audio"
    assert call["pins"][0][0].numel() == 7 * (LATENT_H // 2) * (LATENT_W // 2)
    assert video.shape[1] == 119
    _run(
        pipeline,
        _conditioning("ref2va", 736),
        {"long_video": True, "long_video_mode": "continuation", "h3_seam": {}},
    )
    assert len(calls) == 5 and all(call["seam"] == SeamOptions() and "pins" not in call for call in calls[2:])
    pipeline._pika_spec = _spec("fl2va")
    with pytest.raises(OmniClientError, match="continue_from"):
        _run(pipeline, _conditioning("t2va", 141), {"continue_from": "a", "h3_seam": {}})
    assert len(calls) == 5


def _local_pipeline(task):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module
    from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderMediaConditioning

    pipeline = object.__new__(module.MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.load_text_encoder = True
    pipeline.device = torch.device("cpu")
    pipeline.od_config = None
    pipeline._active_turbo_spec = lambda sampling: None
    pipeline._has_active_native_lora = lambda sampling: False
    pipeline._resolve_task = lambda *args, **kwargs: task
    encoded, encoded_media = [], []

    def encode_text(prepared):
        encoded.append(prepared)
        return torch.full((3, 5120), len(encoded), dtype=torch.bfloat16), torch.zeros(3, dtype=torch.long)

    def encode_media(media):
        encoded_media.append(media)
        return MiniMaxH3EncoderMediaConditioning(
            task=media.task,
            height=media.height,
            width=media.width,
            num_frames=media.num_frames,
            latent_t=media.latent_t,
            audio_t=media.audio_t,
        )

    pipeline.encode_prompt = encode_text
    pipeline._encode_local_media = encode_media
    pipeline._prepare_encoder_conditioning_inputs = lambda conditioning, sampling, **_: {
        "text_embeddings": conditioning.hidden_states,
        "num_frames": conditioning.num_frames,
        "task": conditioning.task,
    }
    return pipeline, encoded, encoded_media


@pytest.mark.parametrize("prompt_count", [7, 8])
def test_window_prompt_count_follows_the_lookahead_plan(prompt_count):
    from PIL import Image

    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    pipeline, encoded, encoded_media = _local_pipeline("ref2va")
    extra = {
        "task": "ref2va",
        "long_video": True,
        "duration": 75,
        "h3_seam": {"lookahead": 17},
        "continuation_prompts": [f"window {i}" for i in range(prompt_count)],
    }
    sampling = OmniDiffusionSamplingParams(width=64, height=64, fps=24, extra_args=extra)
    request = {"prompt": "main prompt", "multi_modal_data": {"image": Image.new("RGB", (256, 256))}}
    if prompt_count == 7:
        with pytest.raises(OmniClientError, match="exactly 8"):
            pipeline._prepare_request_inputs(request, sampling)
        assert not encoded and not encoded_media
        return
    context = pipeline._prepare_request_inputs(request, sampling)
    assert [item.prompt for item in encoded] == extra["continuation_prompts"]
    assert len(context["continuation_text_conditioning"]) == 8


@pytest.mark.parametrize(
    "seam,match", [({"pins": "all"}, "unknown keys: pins"), ({"seed_stride": 10007}, "applies to in-request windows")]
)
def test_rank0_rejects_h3_seam_before_encoding(seam, match):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    pipeline, encoded, encoded_media = _local_pipeline("t2va")
    extra = {"task": "t2va", "duration": 5, "continue_from": "clip-1", "h3_seam": seam}
    sampling = OmniDiffusionSamplingParams(width=64, height=64, fps=24, extra_args=extra)
    with pytest.raises(OmniClientError, match=match):
        pipeline._prepare_request_inputs({"prompt": "main prompt"}, sampling)
    assert not encoded and not encoded_media
    valid = OmniDiffusionSamplingParams(width=64, height=64, fps=24, extra_args={**extra, "h3_seam": {}})
    assert pipeline._prepare_request_inputs({"prompt": "main prompt"}, valid)["task"] == "t2va"
    assert len(encoded) == 1


def _row_keys(packed, text_len):
    positions, tags, used = packed["img_position_ids"], packed["token_tags"], int(packed["cu_seqlens"][1])
    condition = torch.ones(positions.shape[0], dtype=torch.bool)
    condition[packed["img_pos"][packed["update_mask"]]] = False
    condition[packed["audio_pos"][packed["audio_update_mask"]]] = False
    condition[:text_len] = False
    return [(*positions[i].tolist(), int(tags[i]), bool(condition[i])) for i in range(used)]


def test_latent_guide_and_frame0_guide_pack_the_same_rows():
    refs = [{"kind": "image", "latent_h": 4, "latent_w": 4}, {"kind": "audio", "ref_audio_t": 20}]
    guide = {"ref_audio_t": 37, "latent_t": 7, "latent_h": 4, "latent_w": 6}
    common = dict(text_len=5, latent_t=12, latent_h=4, latent_w=6, audio_t=68, temporal_offset=425.0)
    latent_guide = minimax_h3_packed_sequence_ref2va_blocks(
        ref_blocks=[*refs, {"kind": "latent_guide", **guide}], **common
    )
    frame0 = minimax_h3_packed_sequence_ref2va_blocks(
        ref_blocks=refs, guide_blocks=[{"kind": "video_audio", "frame_index": 0, **guide}], **common
    )
    a, b = _row_keys(latent_guide, 5), _row_keys(frame0, 5)
    assert a != b and sorted(a) == sorted(b)
    targets = [[key for key in keys if not key[-1] and key[-2] in (0, 2)] for keys in (a, b)]
    assert targets[0] and targets[0] == targets[1]


def test_diffuse_continuation_without_seam_passes_upstream_arguments():
    def upstream(outputs, kwargs, *, window_frames, overlap_frames, text_conditioning=None, prefix=None):
        origin = prefix.start_frame if prefix is not None else 0
        windows = _upstream_plan(
            kwargs["num_frames"],
            window_frames,
            overlap_frames,
            first_overlap=prefix.overlap_frames if prefix is not None else 0,
            origin=origin,
        )
        media_time_origin = max(item[0].shape[0] for item in text_conditioning) if text_conditioning else None
        video, audio = (prefix.video, prefix.audio) if prefix is not None else (None, None)
        expected = []
        for index, window in enumerate(windows):
            overlap_v = _video_t(window.overlap) if window.overlap else 0
            guided = video is not None and audio is not None
            args = window_args(
                kwargs,
                window,
                video_tail=video[:, :, -overlap_v:] if guided else None,
                audio_tail=audio[..., -window.overlap_audio_t :] if guided else None,
                timeline_guide=prefix is not None and index == 0,
                audio_origin=_audio_t(origin),
            )
            if text_conditioning is not None:
                args["text_embeddings"], args["text_tags"] = text_conditioning[index]
                args["media_time_origin"] = media_time_origin
            expected.append(args)
            sampled_video, sampled_audio = outputs[index]
            if not guided:
                video, audio = sampled_video, sampled_audio
            else:
                video = torch.cat((video, sampled_video[:, :, overlap_v:]), dim=2)
                audio = torch.cat((audio, sampled_audio[..., window.overlap_audio_t :]), dim=-1)
        return expected, video, audio

    def run(kwargs, **options):
        calls, outputs = [], []

        def sample(**kw):
            calls.append(kw)
            generator = torch.Generator().manual_seed(len(calls))
            video = torch.randn(1, 24, kw["latent_t"], kw["latent_h"], kw["latent_w"], generator=generator)
            audio = torch.randn(2, 32, kw["audio_t"], generator=generator)
            outputs.append((video, audio))
            return video, audio

        video, audio = diffuse_continuation(sample, kwargs, window_frames=277, overlap_frames=22, **options)
        expected, expected_video, expected_audio = upstream(
            outputs, kwargs, window_frames=277, overlap_frames=22, **options
        )
        assert len(calls) == len(expected) and torch.equal(video, expected_video) and torch.equal(audio, expected_audio)
        for call, want in zip(calls, expected, strict=True):
            assert not {"seam", "pins", "pin_trace"} & set(call) and call["seed"] == kwargs["seed"]
            assert call.keys() == want.keys()
            for key, value in want.items():
                assert torch.equal(call[key], value) if isinstance(value, torch.Tensor) else call[key] == value, key

    audio_t = _audio_t(736)
    texts = [(torch.full((3 + i, 8), float(i)), torch.ones(3 + i, dtype=torch.long)) for i in range(3)]
    locked = torch.arange(2 * audio_t * 32, dtype=torch.float32).reshape(2 * audio_t, 32)
    run(_kwargs(736, locked_audio_rows=locked), text_conditioning=texts)
    prefix = plan_latent_continuation(
        _clip(124, start=102), ContinuationRequest("a", 22), num_frames=702, latent_h=LATENT_H, latent_w=LATENT_W
    ).prefix
    total = 702
    kwargs = _kwargs(total)
    kwargs["audio_t"] = _audio_t(prefix.start_frame + total) - _audio_t(prefix.start_frame)
    run(kwargs, prefix=prefix)


def test_client_seed_schedule_avoids_correlated_streams():
    forbidden = {0, *PIKA_CORRELATED_SEED_OFFSETS, *(-offset for offset in PIKA_CORRELATED_SEED_OFFSETS)}
    seeds = [base + 10007 * k + 1000003 * r for base in (0, 500000) for k in range(30) for r in range(6)]
    assert not any(b - a in forbidden for i, a in enumerate(seeds) for b in seeds[i + 1 :])
    legacy = [base + 10007 * k + 500000 * r for base in (0, 500000) for k in range(30) for r in range(6)]
    assert any(b - a in forbidden for i, a in enumerate(legacy) for b in legacy[i + 1 :])


def _run_hop(end, seam, *, total=124):
    lookahead = seam.lookahead if seam is not None else 0
    start = end + lookahead - 124
    generator = torch.Generator().manual_seed(end)
    clip = CachedClip(
        video=torch.randn(1, 24, _video_t(124), LATENT_H, LATENT_W, generator=generator),
        audio=torch.randn(2, 32, _audio_t(start + 124) - _audio_t(start), generator=generator),
        start_frame=start,
        num_frames=124,
    )
    plan = plan_latent_continuation(
        clip, ContinuationRequest("a", 22), num_frames=total, latent_h=LATENT_H, latent_w=LATENT_W, lookahead=lookahead
    )
    assert plan.prefix.start_frame + plan.prefix.overlap_frames == end
    kwargs = _kwargs(total, **({} if seam is None else {"seam": seam}))
    kwargs["audio_t"] = plan.audio_t
    calls, outputs = [], []

    def diffuse(**kw):
        calls.append(kw)
        base = 1e6 * len(calls)
        video = base + torch.arange(24 * kw["latent_t"] * LATENT_H * LATENT_W, dtype=torch.float32)
        audio = base + torch.arange(2 * 32 * kw["audio_t"], dtype=torch.float32)
        outputs.append((video.reshape(1, 24, kw["latent_t"], LATENT_H, LATENT_W), audio.reshape(2, 32, kw["audio_t"])))
        return outputs[-1]

    video, audio = diffuse_continuation(diffuse, kwargs, window_frames=277, overlap_frames=22, prefix=plan.prefix)
    assert video.shape[2] == kwargs["latent_t"] and audio.shape[-1] == kwargs["audio_t"]
    return plan.prefix, calls, outputs, video, audio


def _assert_hop_audio(prefix, call, sampled, audio, kept, *, pinned):
    audio_t = call["audio_t"]
    tail = prefix.audio[..., :kept]
    guide = call["guide_blocks"][0]
    assert guide["kind"] == "video_audio" and guide["latent_t"] == 7
    assert guide["ref_audio_t"] == call["audio_condition_lengths"][0] == kept
    assert torch.equal(call["audio_condition"], minimax_h3_pack_audio_latent(tail))
    if pinned:
        rows, values, groups = call["pins"][1]
        steps = torch.arange(kept)
        assert groups == 1 and torch.equal(rows, torch.cat((steps, steps + audio_t)))
        packed = torch.zeros(2 * audio_t, 32)
        packed[rows] = values
        unpacked = minimax_h3_unpack_audio_tokens(packed, audio_t=2 * audio_t, audio_channel=2)
        assert torch.equal(unpacked[..., :kept], tail) and torch.all(unpacked[..., kept:] == 0)
    else:
        assert "pins" not in call
    assert torch.equal(audio[..., :audio_t], torch.cat((tail, sampled[..., kept:]), dim=-1))


@pytest.mark.parametrize("end", [124, 226, 277])
def test_pin_regenerates_the_parent_audio_latent_past_its_last_frame(end):
    prefix, calls, outputs, video, audio = _run_hop(end, SeamOptions(pin="all"))
    assert end % 3 == 1 and prefix.audio.shape[-1] == 37 and len(calls) == 1
    (call,), ((sampled_video, sampled_audio),) = calls, outputs
    _assert_hop_audio(prefix, call, sampled_audio, audio, 36, pinned=True)
    assert call["pins"][0][2] == 7 and call["pins"][0][0].numel() == 7 * (LATENT_H // 2) * (LATENT_W // 2)
    assert audio.shape[-1] == call["audio_t"] == 207
    assert torch.equal(audio, torch.cat((prefix.audio[..., :36], sampled_audio[..., 36:]), dim=-1))
    assert torch.equal(video, torch.cat((prefix.video, sampled_video[:, :, 7:]), dim=2))


def test_pin_drops_only_the_prefix_audio_latent_across_windows():
    prefix, calls, outputs, _, audio = _run_hop(124, SeamOptions(pin="all"), total=702)
    windows = plan_continuation_windows(702, 277, 22, first_overlap=22, origin=102)
    assert len(calls) == len(windows) == 3
    _assert_hop_audio(prefix, calls[0], outputs[0][1], audio, 36, pinned=True)
    bounds = [_audio_t(window.end) - _audio_t(102) for window in windows]
    for index, window in enumerate(windows[1:], 1):
        count, cut = window.overlap_audio_t, bounds[index - 1]
        assert count == 37 and calls[index]["pins"][1][0].numel() == 2 * count
        assert calls[index]["audio_condition_lengths"][-1] == count
        assert torch.equal(calls[index]["audio_condition"], minimax_h3_pack_audio_latent(audio[..., cut - count : cut]))
        assert torch.equal(audio[..., cut : bounds[index]], outputs[index][1][..., count:])
    assert audio.shape[-1] == bounds[-1] == _audio_t(804) - _audio_t(102)


@pytest.mark.parametrize("end,count", [(141, 37), (243, 37), (158, 36), (260, 36)])
def test_pin_keeps_the_prefix_audio_that_ends_inside_the_last_frame(end, count):
    prefix, calls, outputs, _, audio = _run_hop(end, SeamOptions(pin="all"))
    assert end % 3 != 1 and prefix.audio.shape[-1] == count
    _assert_hop_audio(prefix, calls[0], outputs[0][1], audio, count, pinned=True)
    assert audio.shape[-1] == calls[0]["audio_t"]


@pytest.mark.parametrize(
    "seam",
    [
        None,
        SeamOptions(),
        SeamOptions(diag=True),
        SeamOptions(pin="all", lookahead=17),
        SeamOptions(pin="all", lookahead=34),
    ],
)
def test_unpinned_or_lookahead_hops_keep_the_whole_prefix_audio(seam):
    prefix, calls, outputs, _, audio = _run_hop(124, seam)
    assert prefix.audio.shape[-1] == 37
    pinned = seam is not None and seam.pin != "none"
    _assert_hop_audio(prefix, calls[0], outputs[0][1], audio, 37, pinned=pinned)
    assert audio.shape[-1] == calls[0]["audio_t"] == 207


@pytest.mark.parametrize("total", [124, 702])
@pytest.mark.parametrize("end", [124, 141])
def test_chunk_pin_behind_a_prefix_keeps_cumulative_lengths(end, total):
    prefix, calls, outputs, video, audio = _run_hop(end, SeamOptions(pin="chunk"), total=total)
    start = end - 22
    count = _audio_t(start + 17) - _audio_t(start)
    call, (sampled_video, sampled_audio) = calls[0], outputs[0]
    assert call["pins"][0][2] == 5 and call["pins"][1][0].numel() == 2 * count
    assert call["audio_condition_lengths"][0] == prefix.audio.shape[-1] - (end % 3 == 1)
    expected_v = torch.cat((prefix.video[:, :, :5], sampled_video[:, :, 5:]), dim=2)
    expected_a = torch.cat((prefix.audio[..., :count], sampled_audio[..., count:]), dim=-1)
    keep_v, keep_a = call["latent_t"], call["audio_t"]
    if total > 124:
        keep_v, keep_a = keep_v - 7, keep_a - 37
    assert torch.equal(video[:, :, :keep_v], expected_v[:, :, :keep_v])
    assert torch.equal(audio[..., :keep_a], expected_a[..., :keep_a])
    assert (video.shape[2] == keep_v) == (audio.shape[-1] == keep_a) == (total == 124)

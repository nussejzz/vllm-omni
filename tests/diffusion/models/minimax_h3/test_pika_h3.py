# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pika H3 Flex: manifest validation, LoRA key mapping, sigmas, noise streams and the re-noise sampler.

Adapted from the CPU tests of vllm-project/vllm-omni draft PR #9777 to the
request-local sampler framework (``vllm_omni.diffusion.sampler``).
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.minimax_h3 import lora as lora_module
from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline_module
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import minimax_h3_denoise_loop
from vllm_omni.diffusion.models.minimax_h3.lora import load_pika_h3_lora
from vllm_omni.diffusion.models.minimax_h3.pika import (
    PIKA_BASE_REVISION,
    PIKA_KNOTS,
    PIKA_TARGET_MODULES,
    PikaNoise,
    PikaSpec,
    pika_sigmas,
    read_pika_spec,
)
from vllm_omni.diffusion.models.minimax_h3.scheduling_minimax_h3_euler_ancestral import minimax_h3_rf_v_to_x0
from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas
from vllm_omni.diffusion.sampler import (
    RENOISE_SAMPLER,
    RenoiseNoiseSource,
    SamplingState,
    renoise_step,
    resolve_sampler_name,
    sampler_object,
)
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest

from .test_minimax_h3_lora import _TINY_TARGET_DIMS, _write_tiny_turbo
from .test_minimax_h3_step_execution import _make_branch, _make_state, _SegmentMeanModel, _step_pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def _no_pinned_lora_factors_without_cuda(monkeypatch):
    """vLLM pins CPU LoRA factors whenever its platform is CUDA, which it also is
    on a GPU host with ``CUDA_VISIBLE_DEVICES=''``; pinning then fails. These
    are CPU tests, so do not pin when no device is visible."""
    if not torch.cuda.is_available():
        monkeypatch.setattr("vllm.lora.lora_model.PIN_MEMORY", False)


# float32 ladders from the Pika contract (float64 shift, then float32).
_PIKA_VIDEO = [1.0, 0.97297299, 0.92307693, 0.80000001, 0.57142860, 0.33333334, 0.0]
_PIKA_AUDIO = [1.0, 0.89999998, 0.75, 0.5, 0.25, 0.11111111, 0.0]


def _manifest(task="ref2va", digest="0" * 64):
    return {
        "format_version": 1,
        "model": "pika-flex",
        "task": task,
        "release_id": f"pika-flex-{task}-v1",
        "rank": 128,
        "alpha": 128,
        "trunk": "transformer_ref" if task == "ref2va" else "transformer",
        "numerics_policy": "fp32_conditioning_v1" if task == "ref2va" else "bf16_conditioning_v1",
        "sampling": {"knots": list(PIKA_KNOTS), "video_shift": 12.0, "audio_shift": 3.0, "method": "renoise"},
        "base": {"repo_id": "MiniMaxAI/MiniMax-H3", "revision": PIKA_BASE_REVISION},
        "target_modules": list(PIKA_TARGET_MODULES),
        "sha256": digest,
    }


def _artifact(tmp_path, task="ref2va", **kwargs):
    weights = tmp_path / "adapter.safetensors"
    _write_tiny_turbo(weights, **kwargs)
    manifest = _manifest(task, hashlib.sha256(weights.read_bytes()).hexdigest())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return weights


def _load(path, partition="ref2va"):
    return load_pika_h3_lora(
        partition=partition,
        lora_request=LoRARequest("pika", 7, str(path)),
        lora_path=path,
        dtype=torch.bfloat16,
    )


def _spec(task="ref2va"):
    policy = "fp32_conditioning_v1" if task == "ref2va" else "bf16_conditioning_v1"
    return PikaSpec(task, "0" * 64, f"pika-flex-{task}-v1", policy, "/unused/adapter.safetensors")


# ---------------------------------------------------------------- sigmas


def test_pika_sigmas_are_float32_contract_values():
    video, audio = pika_sigmas(12.0), pika_sigmas(3.0)
    assert len(video) == len(audio) == 7
    torch.testing.assert_close(torch.tensor(video), torch.tensor(_PIKA_VIDEO), rtol=0, atol=1e-8)
    torch.testing.assert_close(torch.tensor(audio), torch.tensor(_PIKA_AUDIO), rtol=0, atol=1e-8)
    for value in video + audio:
        assert torch.tensor(value, dtype=torch.float32).item() == value  # exactly float32
    assert _spec().video_sigmas == video and _spec().audio_sigmas == audio
    assert _spec().base_schedule == tuple(k / 1000 for k in PIKA_KNOTS)
    # The generic base_schedule path shifts in Python float64 and keeps doubles:
    # not the float32 grid Pika feeds to the model.
    generic = minimax_h3_time_shift_sigmas(num_steps=6, shift_scale=12.0, base_schedule=_spec().base_schedule)
    assert any(a != b for a, b in zip(generic, video))


def test_grid_sigma_x0_differs_from_timestep_round_trip():
    """x0 must use the grid sigma; 1 - float32(1 - sigma) is one ulp off at some knots."""
    differing = []
    for sigma in pika_sigmas(12.0)[1:-1] + pika_sigmas(3.0)[1:-1]:
        t = torch.tensor(1.0 - sigma, dtype=torch.float32)
        x, v = torch.zeros(1), torch.ones(1)
        grid = x + sigma * v
        assert grid.item() == sigma
        if minimax_h3_rf_v_to_x0(x, v, t).item() != sigma:
            differing.append(sigma)
    assert differing  # the base round trip would not reproduce Pika's x0


# ---------------------------------------------------------------- manifest / loader


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_pika_loads_all_factors_and_preserves_alpha_and_gate_order(tmp_path, monkeypatch, task):
    monkeypatch.setattr(lora_module, "_TURBO_TARGET_DIMS", _TINY_TARGET_DIMS)
    path = _artifact(tmp_path, task)
    model, peft, spec = _load(path, task)
    # 50 blocks + 2 token-refiner blocks, 6 targets each; q/k/v stay separate targets here.
    assert len(model.loras) == 312
    assert peft.r == peft.lora_alpha == 128
    assert spec.task == task and spec.numerics_policy == _manifest(task)["numerics_policy"]
    assert "token_refiner.blocks.0.attn.to_q" in model.loras
    assert "blocks.49.attn.out_proj" in model.loras
    fc1 = model.get_lora("blocks.0.mlp.fc1")
    assert all(x.dtype == torch.bfloat16 for x in fc1.lora_a + fc1.lora_b)
    assert all(x == 1.0 for x in fc1.scaling)
    # Diffusers ff.net.0.proj is [value; gate]; H3's fused fc1 is [gate; up].
    torch.testing.assert_close(fc1.lora_b[0], torch.full((2, 128), 2, dtype=torch.bfloat16))
    torch.testing.assert_close(fc1.lora_b[1], torch.ones(2, 128, dtype=torch.bfloat16))
    assert read_pika_spec(tmp_path) == read_pika_spec(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("format_version", 2),
        ("rank", 64),
        ("alpha", 8),
        ("trunk", "transformer"),
        ("numerics_policy", "bf16_conditioning_v1"),
        ("release_id", "pika-flex-ref2va-v2"),
        ("base", {"repo_id": "MiniMaxAI/MiniMax-H3", "revision": "0" * 40}),
        ("sampling", {"knots": list(PIKA_KNOTS), "video_shift": 12.0, "audio_shift": 3.0, "method": "euler"}),
        ("sampling", {"knots": [1000, 750, 500, 250, 0], "video_shift": 12.0, "audio_shift": 3.0, "method": "renoise"}),
        ("target_modules", ["to_q"]),
        ("sha256", "invalid"),
        ("task", "t2va"),
    ],
)
def test_pika_rejects_modified_contract(tmp_path, field, value):
    _artifact(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest[field] = value
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_pika_spec(tmp_path)


def test_pika_rejects_bad_hash_wrong_partition_and_missing_factors(tmp_path, monkeypatch):
    monkeypatch.setattr(lora_module, "_TURBO_TARGET_DIMS", _TINY_TARGET_DIMS)
    path = _artifact(tmp_path, omit_target="transformer_blocks.49.attn.to_q")
    with pytest.raises(ValueError, match="target set"):
        _load(path)
    with pytest.raises(ValueError, match="dedicated"):
        _load(path, "fl2va")
    with pytest.raises(ValueError, match="BF16"):
        load_pika_h3_lora(
            partition="ref2va", lora_request=LoRARequest("p", 7, str(path)), lora_path=path, dtype=torch.float32
        )
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        _load(path)


def test_pika_does_not_claim_unrelated_adapter_or_allow_ambiguous_root(tmp_path):
    save_file({"x": torch.ones(1)}, str(tmp_path / "adapter.safetensors"))
    assert _load(tmp_path) is None
    assert read_pika_spec(tmp_path / "adapter.safetensors") is None
    (tmp_path / "manifest.json").write_text(json.dumps({"model": "other"}))
    assert read_pika_spec(tmp_path) is None
    root = tmp_path / "release"
    (root / "ref2va").mkdir(parents=True)
    (root / "ref2va" / "manifest.json").write_text(json.dumps(_manifest()))
    with pytest.raises(ValueError, match="directory"):
        read_pika_spec(root)


# ---------------------------------------------------------------- request contract


def _sampling(**overrides):
    return OmniDiffusionSamplingParams(**{"num_inference_steps": None, "extra_args": {}, **overrides})


_REJECTED_OVERRIDES = [
    {"num_inference_steps": 7},
    {"num_inference_steps": 50},
    {"extra_args": {"flow_shift": 6}},
    {"extra_args": {"audio_flow_shift": float("nan")}},
    {"extra_args": {"audio_flow_shift": "x"}},
    {"extra_args": {"long_video_mode": "continuation"}},
    {"extra_args": {"continuation_prompts": ["a", "b"]}},
    {"extra_args": {"audio_mode": "lock_source"}},
    {"extra_args": {"sampler": "res_multistep"}},
    {"extra_args": {"continue_from": "clip-123"}},
    {"extra_args": {"regenerate": True}},
    {"quality": "high"},
    {"lora_request": LoRARequest("other", 3, "/unused")},
    {"num_outputs_per_prompt": 2},
    {"num_outputs_per_prompt": 10},
]


@pytest.mark.parametrize("overrides", _REJECTED_OVERRIDES)
def test_pika_request_cannot_silently_change_sampling(overrides):
    with pytest.raises(ValueError):
        _spec().check_request(_sampling(**overrides), "ref2va")


_ACCEPTED_OVERRIDES = [
    {},
    {"num_inference_steps": 6},
    {"extra_args": {"flow_shift": "12", "audio_flow_shift": 3}},
    {"quality": "lossless"},
    {"num_outputs_per_prompt": 1},
]


@pytest.mark.parametrize("overrides", _ACCEPTED_OVERRIDES)
def test_pika_request_accepts_its_own_contract(overrides):
    _spec().check_request(_sampling(**overrides), "ref2va")


def test_pika_rejects_multiple_outputs_because_adjacent_seeds_share_streams():
    """Output k is seeded ``seed + k``: its initial noise would be output k-1's step-0 re-noise."""
    from vllm_omni.diffusion.models.minimax_h3.pika import PIKA_CORRELATED_SEED_OFFSETS

    assert pipeline_module._minimax_h3_output_seeds(42, 2) == [42, 43]
    first, second = _noise(42), _noise(43)
    renoise0_video, renoise0_audio = first.draw(0, (torch.zeros(12, 96), torch.zeros(6, 32)))
    initial_video, initial_audio = second.initial_rows()
    assert torch.equal(initial_video, renoise0_video) and torch.equal(initial_audio, renoise0_audio)
    assert PIKA_CORRELATED_SEED_OFFSETS == (1, 7918, 7919)
    with pytest.raises(ValueError, match="one output per request"):
        _spec().check_request(_sampling(num_outputs_per_prompt=2), "ref2va")
    pipe = object.__new__(pipeline_module.MiniMaxH3Pipeline)
    pipe.od_config = OmniDiffusionConfig()
    pipe._pika_spec = _spec()
    with pytest.raises(OmniClientError, match="one output per request"):
        pipe._validate_pika_request(_sampling(num_outputs_per_prompt=2), "ref2va")


def test_pika_task_and_step_execution_rules():
    with pytest.raises(ValueError, match="task"):
        _spec().check_request(_sampling(), "fl2va")
    with pytest.raises(ValueError, match="explicit"):
        _spec().check_request(_sampling(), "ref2va", step_execution=True)
    _spec().check_request(_sampling(num_inference_steps=6), "ref2va", step_execution=True)


def test_pika_sigma_positions_and_pipeline_request_validation():
    pipe = object.__new__(pipeline_module.MiniMaxH3Pipeline)
    pipe.od_config = OmniDiffusionConfig()
    pipe._fasth3 = None
    pipe._pika_spec = _spec()
    positions, steps = pipe._resolve_sigma_positions("ref2va", _sampling(num_inference_steps=50))
    assert steps == 6 and positions == (1.0, 0.75, 0.5, 0.25, 0.1, 0.04, 0.0)
    assert pipe._validate_pika_request(_sampling(), "ref2va") is pipe._pika_spec
    with pytest.raises(OmniClientError):
        pipe._validate_pika_request(_sampling(num_inference_steps=4), "ref2va")
    base = object.__new__(pipeline_module.MiniMaxH3Pipeline)
    base.od_config = OmniDiffusionConfig()
    base._pika_spec = None
    assert base._validate_pika_request(_sampling(num_inference_steps=4), "ref2va") is None


def _startup_pipe(partition="ref2va", base_schedule=None):
    pipe = object.__new__(pipeline_module.MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipe)
    pipe.partition = partition
    pipe._base_schedule_by_partition = {partition: base_schedule}
    return pipe


def test_pika_startup_mode_is_claimed_only_by_a_pika_lora_path(tmp_path):
    _artifact(tmp_path)
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        pipe = _startup_pipe()
        spec = pipe._resolve_startup_pika(SimpleNamespace(lora_path=str(tmp_path), lora_scale=1.0), modular=False)
        assert spec is not None and spec.task == "ref2va"
        assert pipe.diffusion_lora_fixed_adapter is True
        assert torch.backends.cuda.matmul.allow_tf32 is False
        # Base servers: no --lora-path or a non-Pika one leave everything unchanged.
        base = _startup_pipe()
        assert base._resolve_startup_pika(SimpleNamespace(lora_path=None), modular=False) is None
        assert not hasattr(base, "diffusion_lora_fixed_adapter") or base.diffusion_lora_fixed_adapter is not True
        other = tmp_path / "other"
        other.mkdir()
        save_file({"x": torch.ones(1)}, str(other / "adapter.safetensors"))
        assert base._resolve_startup_pika(SimpleNamespace(lora_path=str(other)), modular=False) is None
        with pytest.raises(ValueError, match="task-type"):
            _startup_pipe("fl2va")._resolve_startup_pika(SimpleNamespace(lora_path=str(tmp_path)), modular=False)
        with pytest.raises(ValueError, match="scale"):
            _startup_pipe()._resolve_startup_pika(
                SimpleNamespace(lora_path=str(tmp_path), lora_scale=0.5), modular=False
            )
        with pytest.raises(ValueError, match="dynamic"):
            _startup_pipe()._resolve_startup_pika(
                SimpleNamespace(lora_path=str(tmp_path), lora_application="merged"), modular=False
            )
        with pytest.raises(ValueError, match="original"):
            _startup_pipe()._resolve_startup_pika(SimpleNamespace(lora_path=str(tmp_path)), modular=True)
        fasth3 = _startup_pipe()
        fasth3._fasth3_checkpoint = object()
        with pytest.raises(ValueError, match="original"):
            fasth3._resolve_startup_pika(SimpleNamespace(lora_path=str(tmp_path)), modular=False)
        for backend in ("cache_dit", "tea_cache"):
            with pytest.raises(ValueError, match="cache-backend none"):
                _startup_pipe()._resolve_startup_pika(
                    SimpleNamespace(lora_path=str(tmp_path), cache_backend=backend), modular=False
                )
        assert (
            _startup_pipe()._resolve_startup_pika(
                SimpleNamespace(lora_path=str(tmp_path), cache_backend="none"), modular=False
            )
            is not None
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def test_pika_debug_base_dit_does_not_pin_the_adapter(tmp_path, monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.pika import PIKA_DEBUG_BASE_DIT_ENV

    _artifact(tmp_path)
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        monkeypatch.setenv(PIKA_DEBUG_BASE_DIT_ENV, "1")
        pipe = _startup_pipe()
        assert pipe._resolve_startup_pika(SimpleNamespace(lora_path=str(tmp_path)), modular=False) is not None
        assert getattr(pipe, "diffusion_lora_fixed_adapter", False) is not True
        monkeypatch.setenv(PIKA_DEBUG_BASE_DIT_ENV, "0")
        pipe = _startup_pipe()
        pipe._resolve_startup_pika(SimpleNamespace(lora_path=str(tmp_path)), modular=False)
        assert pipe.diffusion_lora_fixed_adapter is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("attribute", ["_vdn", "_regenerate", "_fasth3", "_fasth3_checkpoint"])
def test_pika_server_mode_is_exclusive_with_other_lora_path_consumers(attribute):
    pipe = _startup_pipe()
    pipe._pika_spec = None
    setattr(pipe, attribute, object())
    pipe._check_pika_is_exclusive()  # base servers: unchanged
    pipe._pika_spec = _spec()
    with pytest.raises(ValueError, match="cannot be combined"):
        pipe._check_pika_is_exclusive()
    setattr(pipe, attribute, None)
    pipe._check_pika_is_exclusive()


def test_pika_adapter_only_loads_as_the_startup_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(lora_module, "_TURBO_TARGET_DIMS", _TINY_TARGET_DIMS)
    path = _artifact(tmp_path)
    pipe = _startup_pipe()
    pipe._turbo_lora_specs, pipe._native_distilled_lora_adapter_ids, pipe._lora_sigma_schedules = {}, set(), {}
    pipe._pika_lora_specs = {}
    pipe._pika_spec = None
    request = LoRARequest("pika", 7, str(tmp_path))
    with pytest.raises(ValueError, match="startup"):
        pipe._load_diffusion_lora_adapter(lora_request=request, lora_path=str(tmp_path), dtype=torch.bfloat16)
    pipe._pika_spec = read_pika_spec(path)
    model, _ = pipe._load_diffusion_lora_adapter(lora_request=request, lora_path=str(tmp_path), dtype=torch.bfloat16)
    assert pipe._pika_lora_specs[7] == pipe._pika_spec
    with pytest.raises(ValueError, match="incomplete"):
        pipe._validate_diffusion_lora_binding(lora_model=model, bound_lora_names=frozenset())
    pipe._validate_diffusion_lora_binding(lora_model=model, bound_lora_names=frozenset(model.loras))


# ---------------------------------------------------------------- noise streams


def _noise(seed=42):
    return PikaNoise(seed=seed, device=torch.device("cpu"), latent_t=2, latent_h=4, latent_w=6, audio_t=3)


def _video_rows(external):
    # Independent explicit packing of [1,T,24,H,W] into [T*H/2*W/2, 96] (c, ph, pw within a row).
    _, t, c, h, w = external.shape
    return external.reshape(t, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(-1, 96)


def _audio_rows(external):
    # [1, T, 64] channel-major: all left rows, then all right rows.
    return torch.cat((external[0, :, :32], external[0, :, 32:]), dim=0)


def test_pika_noise_matches_reference_external_layout_and_isolates_rng():
    global_state = torch.random.get_rng_state().clone()
    noise = _noise()
    reference = torch.Generator().manual_seed(42)
    video = torch.randn(1, 2, 24, 4, 6, generator=reference)
    audio = torch.randn(1, 3, 64, generator=reference)
    initial_video, initial_audio = noise.initial_rows()
    torch.testing.assert_close(initial_video, _video_rows(video), rtol=0, atol=0)
    torch.testing.assert_close(initial_audio, _audio_rows(audio), rtol=0, atol=0)

    renoise = torch.Generator().manual_seed(43)
    like = (initial_video, initial_audio)
    for step in range(2):
        expected = (
            _video_rows(torch.randn(1, 2, 24, 4, 6, generator=renoise)),
            _audio_rows(torch.randn(1, 3, 64, generator=renoise)),
        )
        drawn = noise.draw(step, like)
        replay = noise.draw(step, like)  # a retried step replays its draws
        for a, b, c in zip(drawn, replay, expected, strict=True):
            torch.testing.assert_close(a, c, rtol=0, atol=0)
            torch.testing.assert_close(b, c, rtol=0, atol=0)
    before = noise.renoise.get_state().clone()
    assert noise.draw(2, (None, None)) == (None, None)  # terminal knot: no draw
    assert torch.equal(before, noise.renoise.get_state())
    assert torch.equal(global_state, torch.random.get_rng_state())


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_pika_condition_precision_and_fixed_augmentation(task):
    spec = _spec(task)
    rows = torch.linspace(-1.001, 1.001, 2 * 6 * 96).reshape(12, 96)
    shapes = [(1, 4, 6), (1, 4, 6)]
    actual = _noise().visual_anchor(rows, shapes, spec.conditioning_dtype)
    generator = torch.Generator().manual_seed(42 + 7919)
    eps = torch.cat([_video_rows(torch.randn(1, 1, 24, 4, 6, generator=generator)) for _ in shapes])
    clean = rows.to(spec.conditioning_dtype).float()
    expected = 0.999 * clean + (1.0 - 0.999) * eps
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert spec.conditioning_dtype == (torch.float32 if task == "ref2va" else torch.bfloat16)
    with pytest.raises(ValueError, match="do not match"):
        _noise().visual_anchor(rows, [(1, 4, 6)], spec.conditioning_dtype)


# ---------------------------------------------------------------- sampler


def test_renoise_is_internal_and_matches_pika_add_noise():
    with pytest.raises(ValueError):
        resolve_sampler_name(RENOISE_SAMPLER)  # requests cannot select it
    assert sampler_object(RENOISE_SAMPLER) is not None
    x0 = torch.randn(5, 7)
    eps = torch.randn(5, 7)
    for sigma in pika_sigmas(12.0)[1:-1] + pika_sigmas(3.0)[1:-1]:
        s = torch.tensor(sigma, dtype=torch.float32).reshape(1, 1)
        expected = ((1 - s) * x0.float() + s * eps.float()).to(x0.dtype)  # Pika sampling.py:14-21
        torch.testing.assert_close(renoise_step(x0, eps, sigma_next=sigma), expected, rtol=0, atol=0)
    assert renoise_step(x0, None, sigma_next=0.0) is x0


class _CountingSource(RenoiseNoiseSource):
    def __init__(self):
        self.calls = []

    def draw(self, step_index, like):
        self.calls.append((step_index, tuple(x is not None for x in like)))
        return tuple(None if x is None else torch.zeros_like(x) for x in like)


def test_renoise_sampler_requires_source_and_skips_terminal_draw():
    state = SamplingState((torch.ones(2), torch.ones(3)), (pika_sigmas(12.0), pika_sigmas(3.0)))
    with pytest.raises(ValueError, match="RenoiseNoiseSource"):
        sampler_object(RENOISE_SAMPLER).sample(lambda es: [tuple(e.latents) for e in es], [state])
    source = _CountingSource()
    state = SamplingState((torch.ones(2), torch.ones(3)), (pika_sigmas(12.0), pika_sigmas(3.0)), history=source)
    sampler_object(RENOISE_SAMPLER).sample(lambda es: [tuple(x * 2 for x in e.latents) for e in es], [state])
    assert state.finished and state.history is source
    assert source.calls == [(i, (True, True)) for i in range(5)] + [(5, (False, False))]


def _synthetic_x0(x, sigma):
    """Elementwise stand-in denoiser, identical in any layout."""
    return 0.5 * x + float(sigma) * 0.25 - 0.125


def test_renoise_sampler_matches_pika_few_step_sample_bitwise():
    """sample_renoise + PikaNoise in row layout == Pika's few_step_sample in external layout."""
    seed = 1234
    sv, sa = pika_sigmas(12.0), pika_sigmas(3.0)
    # --- Pika reference (src/pika_h3/core/sampling.py:87-109,138-169), external layout
    initial = torch.Generator().manual_seed(seed)
    video = torch.randn(1, 2, 24, 4, 6, generator=initial)
    audio = torch.randn(1, 3, 64, generator=initial)
    renoise = torch.Generator().manual_seed(seed + 1)
    for i in range(5):
        pred_v, pred_a = _synthetic_x0(video, sv[i]), _synthetic_x0(audio, sa[i])
        eps = torch.randn(pred_v.shape, dtype=torch.float32, generator=renoise)
        s = torch.tensor(sv[i + 1]).reshape(1, 1, 1, 1, 1)
        video = ((1 - s) * pred_v.float() + s * eps.float()).to(pred_v.dtype)
        eps = torch.randn(pred_a.shape, dtype=torch.float32, generator=renoise)
        s = torch.tensor(sa[i + 1]).reshape(1, 1, 1)
        audio = ((1 - s) * pred_a.float() + s * eps.float()).to(pred_a.dtype)
    video, audio = _synthetic_x0(video, sv[5]), _synthetic_x0(audio, sa[5])
    # --- vLLM-Omni sampler in packed rows
    noise = PikaNoise(seed=seed, device=torch.device("cpu"), latent_t=2, latent_h=4, latent_w=6, audio_t=3)
    state = SamplingState(noise.initial_rows(), (sv, sa), history=noise)
    model = lambda es: [tuple(_synthetic_x0(x, s) for x, s in zip(e.latents, e.sigmas)) for e in es]  # noqa: E731
    sampler_object(RENOISE_SAMPLER).sample(model, [state])
    torch.testing.assert_close(state.latents[0], _video_rows(video), rtol=0, atol=0)
    torch.testing.assert_close(state.latents[1], _audio_rows(audio), rtol=0, atol=0)


def _pika_loop(model, branch, video, audio, seed):
    return minimax_h3_denoise_loop(
        model=model,
        positive=branch,
        initial_video_rows=video,
        initial_audio_rows=audio,
        keyframe_cond_rows=None,
        sigmas_video=pika_sigmas(12.0),
        sigmas_audio=pika_sigmas(3.0),
        device=torch.device("cpu"),
        sampler=RENOISE_SAMPLER,
        noise_source=_noise(seed),
    )


def test_denoise_loop_requires_noise_source_for_renoise():
    model = _SegmentMeanModel()
    branch, video, audio = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=1)
    with pytest.raises(ValueError, match="noise source"):
        minimax_h3_denoise_loop(
            model=model,
            positive=branch,
            initial_video_rows=video,
            initial_audio_rows=audio,
            keyframe_cond_rows=None,
            sigmas_video=pika_sigmas(12.0),
            sigmas_audio=pika_sigmas(3.0),
            device=torch.device("cpu"),
            sampler=RENOISE_SAMPLER,
        )


class _RecordingModel(_SegmentMeanModel):
    def __call__(self, **kwargs):
        video, audio = super().__call__(**kwargs)
        self.calls[-1]["video_out"], self.calls[-1]["audio_out"] = video.clone(), audio.clone()
        return video, audio


def test_denoise_loop_uses_grid_sigma_x0_and_fresh_noise():
    """Request mode: six forwards, x0 = x + sigma_grid * v, re-noise from seed + 1, video before audio."""
    model = _RecordingModel()
    branch, video, audio = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=3)
    assert bool(branch.update_mask.all()) and bool(branch.audio_update_mask.all())
    out_video, out_audio = _pika_loop(model, branch, video, audio, seed=3)
    assert len(model.calls) == 6
    reference = _noise(3)
    sv, sa = pika_sigmas(12.0), pika_sigmas(3.0)
    rows_v, rows_a = video.clone(), audio.clone()
    for i, call in enumerate(model.calls):
        torch.testing.assert_close(call["video_rows"], rows_v, rtol=0, atol=0)
        torch.testing.assert_close(call["audio_rows"], rows_a, rtol=0, atol=0)
        # AdaLN time is 1 - sigma written into float32.
        assert call["video_timesteps"][0].item() == torch.tensor(1.0 - sv[i], dtype=torch.float32).item()
        x0_v = rows_v + sv[i] * call["video_out"].float()
        x0_a = rows_a + sa[i] * call["audio_out"].float()
        eps = reference.draw(i, (x0_v if sv[i + 1] else None, x0_a if sa[i + 1] else None))
        rows_v = renoise_step(x0_v, eps[0], sigma_next=sv[i + 1])
        rows_a = renoise_step(x0_a, eps[1], sigma_next=sa[i + 1])
    torch.testing.assert_close(out_video, rows_v, rtol=0, atol=0)
    torch.testing.assert_close(out_audio, rows_a, rtol=0, atol=0)


def _pika_state(model, branch, video, audio, seed, request_id):
    state = _make_state(request_id, model, branch, video, audio, pika_sigmas(12.0), pika_sigmas(3.0), RENOISE_SAMPLER)
    sample = state.extra[pipeline_module._STEP_SAMPLING_STATE]
    sample.history = _noise(seed)
    return state


def test_pika_step_execution_matches_request_mode_including_interleaving_and_cobatching():
    model = _SegmentMeanModel()
    expected, solo_states, batch_states = [], [], []
    for seed in (7, 99):
        branch, video, audio = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=seed)
        expected.append(_pika_loop(model, branch, video, audio, seed))
        solo_states.append(_pika_state(model, branch, video, audio, seed, f"solo-{seed}"))
        batch_states.append(_pika_state(model, branch, video, audio, seed, f"batch-{seed}"))
    pipe = _step_pipeline(model)
    for _ in range(6):  # interleaved, one request per call, reversed order
        for state in reversed(solo_states):
            assert pipe.denoise_step(None, states=[state]) == {state.request_id: None}
    for _ in range(6):  # co-batched into one packed forward
        results = pipe.denoise_step(None, states=batch_states)
        assert all(value is None for value in results.values())
    for states in (solo_states, batch_states):
        for state, (video, audio) in zip(states, expected, strict=True):
            assert state.denoise_completed and state.step_index == 6
            torch.testing.assert_close(state.latents, video, rtol=0, atol=0)
            torch.testing.assert_close(state.extra[pipeline_module._STEP_AUDIO_ROWS], audio, rtol=0, atol=0)


def test_pika_request_preparation_uses_release_noise_and_pins_anchors():
    model = _SegmentMeanModel()
    pipe = _step_pipeline(model)
    spec = _spec()
    clean = torch.linspace(-1.001, 1.001, 6 * 96).reshape(6, 96)
    inputs = pipe._build_denoise_inputs(
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
        visual_condition=clean,
        visual_condition_shape=(1, 4, 6),
        audio_condition=None,
        ref_audio_t=None,
        ref_blocks=[{"kind": "image", "latent_h": 4, "latent_w": 6}],
        pika=spec,
        # The base runtime can prepare CPU noise ahead of encoding; Pika ignores it.
        precomputed_initial_noise=(torch.zeros(12, 96), torch.zeros(6, 32)),
        precomputed_visual_condition_noise=torch.zeros_like(clean),
    )
    branch = inputs["branch"]
    expected = _noise()
    initial_video, initial_audio = expected.initial_rows()
    torch.testing.assert_close(inputs["video_rows"][branch.update_mask], initial_video, rtol=0, atol=0)
    torch.testing.assert_close(inputs["audio_rows"], initial_audio, rtol=0, atol=0)
    anchor = expected.visual_anchor(clean, [(1, 4, 6)], spec.conditioning_dtype)
    torch.testing.assert_close(inputs["cond_anchor"], anchor, rtol=0, atol=0)
    assert inputs["sigmas_video"] == pika_sigmas(12.0)
    assert inputs["sigmas_audio"] == pika_sigmas(3.0)
    assert inputs["sampler"] == RENOISE_SAMPLER and isinstance(inputs["noise_source"], PikaNoise)
    video, _ = minimax_h3_denoise_loop(
        model=model,
        positive=branch,
        initial_video_rows=inputs["video_rows"],
        initial_audio_rows=inputs["audio_rows"],
        keyframe_cond_rows=inputs["cond_anchor"],
        sigmas_video=inputs["sigmas_video"],
        sigmas_audio=inputs["sigmas_audio"],
        device=torch.device("cpu"),
        sampler=inputs["sampler"],
        noise_source=inputs["noise_source"],
    )
    assert len(model.calls) == 6
    torch.testing.assert_close(video[~branch.update_mask], anchor, rtol=0, atol=0)
    for call in model.calls:
        torch.testing.assert_close(call["video_rows"][~branch.update_mask], anchor, rtol=0, atol=0)
    # Unsupported conditioning is rejected before any noise is drawn.
    with pytest.raises(OmniClientError, match="does not support"):
        pipe._build_denoise_inputs(
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
            visual_condition=clean,
            visual_condition_shape=(1, 4, 6),
            audio_condition=None,
            ref_audio_t=None,
            ref_blocks=[{"kind": "image", "latent_h": 4, "latent_w": 6}],
            locked_audio_rows=torch.zeros(6, 32),
            pika=spec,
        )


def test_base_build_inputs_unchanged_without_pika():
    """No pika: CPU noise, Euler, float64 base-schedule path exactly as before."""
    model = _SegmentMeanModel()
    pipe = _step_pipeline(model)
    inputs = pipe._build_denoise_inputs(
        task="t2va",
        text_embeddings=torch.ones(3, 8),
        text_tags=torch.ones(3, dtype=torch.long),
        seed=5,
        latent_t=2,
        latent_h=4,
        latent_w=6,
        audio_t=3,
        num_frames=22,
        num_steps=4,
        video_shift=12.0,
        audio_shift=3.0,
        base_schedule=None,
        visual_condition=None,
        visual_condition_shape=None,
        audio_condition=None,
        ref_audio_t=None,
    )
    assert inputs["noise_source"] is None and inputs["sampler"] == "euler"
    video, audio = pipe._initial_noise(seed=5, latent_t=2, latent_h=4, latent_w=6, audio_t=3)
    torch.testing.assert_close(inputs["video_rows"], video, rtol=0, atol=0)
    torch.testing.assert_close(inputs["audio_rows"], audio, rtol=0, atol=0)
    assert inputs["sigmas_video"] == minimax_h3_time_shift_sigmas(num_steps=4, shift_scale=12.0)


# ---------------------------------------------------------------- parity tooling / opt-in numerics


def test_noise_replication_guard_is_a_noop_on_one_rank():
    pipeline_module._check_replicated_across_dit_ranks(torch.ones(3), torch.zeros(2), what="noise")


def test_golden_conditioning_injection_checks_presentation(tmp_path):
    from vllm_omni.diffusion.models.minimax_h3.pika import load_pika_golden_conditioning

    latents = torch.randn(1, 1, 24, 4, 6)
    torch.save(
        {
            "prompt_embeds": torch.ones(1, 3, 8, dtype=torch.bfloat16),
            "text_token_tags": torch.tensor([1, 0, 1]),
            "references": [{"kind": "image", "video_latents": latents}],
        },
        tmp_path / "03_cond.pt",
    )
    text, rows = load_pika_golden_conditioning(
        str(tmp_path), text_tags=torch.tensor([1, 0, 1]), visual_condition=torch.zeros(6, 96)
    )
    assert text.shape == (3, 8)
    torch.testing.assert_close(rows, _video_rows(latents), rtol=0, atol=0)
    with pytest.raises(ValueError, match="text presentation"):
        load_pika_golden_conditioning(str(tmp_path), text_tags=torch.tensor([1, 1, 1]), visual_condition=None)
    with pytest.raises(ValueError, match="visual rows"):
        load_pika_golden_conditioning(
            str(tmp_path), text_tags=torch.tensor([1, 0, 1]), visual_condition=torch.zeros(12, 96)
        )


def test_debug_dump_is_rank0_and_opt_in(tmp_path):
    from vllm_omni.diffusion.models.minimax_h3.pika import PikaDebugDump

    assert not PikaDebugDump(None, rank=0, seed=1)
    assert not PikaDebugDump(str(tmp_path), rank=1, seed=1)
    dump = PikaDebugDump(str(tmp_path), rank=0, seed=1)
    dump.save("x", value=torch.ones(2), meta=3)
    saved = torch.load(next(tmp_path.glob("req_*_s1")) / "x.pt", weights_only=False)
    assert saved["meta"] == 3 and torch.equal(saved["value"], torch.ones(2))


def test_vae_fp32_flags_are_opt_in(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module

    seen = []

    class FakeRemote(torch.nn.Module):
        def __init__(self):
            super().__init__()
            seen.append(torch.get_default_dtype())
            self.model = torch.nn.Module()
            self.model.decoder = torch.nn.Linear(1, 1)

    monkeypatch.setattr(
        vae_module,
        "_load_component_config",
        lambda _path: {"latent_channels": 1, "latents_mean": [0.0], "latents_std": [1.0]},
    )
    monkeypatch.setattr(vae_module, "_load_remote_component", lambda _p, _c, *, trust_remote_code: FakeRemote())
    installed = []
    monkeypatch.setattr(vae_module, "install_h3_vae_optimizations", lambda decoder, device: installed.append(decoder))
    monkeypatch.setattr(vae_module, "install_temporal_stream_patches", lambda *a, **k: None)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        default = vae_module.MiniMaxH3VideoVAE("unused", device=torch.device("cpu"))
        fp32 = vae_module.MiniMaxH3VideoVAE("unused", device=torch.device("cpu"), build_fp32=True, decode_fp32=True)
    finally:
        torch.set_default_dtype(previous)
    assert seen == [torch.bfloat16, torch.float32]
    assert default.decode_dtype == torch.float16 and len(installed) == 1  # base: FP16 autocast + fused path
    assert fp32.decode_dtype == torch.float32  # Pika: FP32 decode, fused FP16 path skipped
    assert default.keyframe_encode_cudnn_tf32 is None  # base keeps its Hopper FP32 image-encode policy


def test_pika_image_encode_policy_allows_cudnn_tf32():
    """Pika encodes image references with cuDNN TF32 on; base keeps its own policy."""
    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module

    saved = (torch.backends.cudnn.allow_tf32, torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        torch.backends.cudnn.allow_tf32, torch.backends.cudnn.deterministic = False, True
        with vae_module._minimax_h3_keyframe_encode_context(torch.device("cuda", 0), cudnn_tf32=True):
            assert torch.backends.cudnn.allow_tf32 is True
            assert torch.backends.cudnn.deterministic is False and torch.backends.cudnn.benchmark is False
        assert (torch.backends.cudnn.allow_tf32, torch.backends.cudnn.deterministic) == (False, True)
        with vae_module._minimax_h3_keyframe_encode_context(torch.device("cpu"), cudnn_tf32=True):
            assert torch.backends.cudnn.allow_tf32 is False  # CPU: untouched
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = saved


# ---------------------------------------------------------------- fl2va adapter (t2va requests)


def test_pika_request_tasks_per_adapter():
    from vllm_omni.diffusion.models.minimax_h3.pika import PIKA_REQUEST_TASKS, PIKA_SERVED_TASKS

    assert PIKA_SERVED_TASKS == frozenset({"ref2va", "fl2va"})
    assert PIKA_REQUEST_TASKS == {"ref2va": frozenset({"ref2va"}), "fl2va": frozenset({"t2va"})}
    assert _spec("ref2va").request_tasks == frozenset({"ref2va"})
    assert _spec("fl2va").request_tasks == frozenset({"t2va"})


def test_pika_fl2va_adapter_serves_t2va_and_rejects_keyframes():
    spec = _spec("fl2va")
    spec.check_request(_sampling(), "t2va")
    spec.check_request(_sampling(num_inference_steps=6), "t2va", step_execution=True)
    with pytest.raises(ValueError, match="explicit"):
        spec.check_request(_sampling(), "t2va", step_execution=True)
    with pytest.raises(ValueError, match="not ported"):
        spec.check_request(_sampling(), "fl2va")  # H3 first/last-frame keyframes
    with pytest.raises(ValueError, match="fl2va adapter cannot serve task=ref2va"):
        spec.check_request(_sampling(), "ref2va")
    # The ref2va adapter keeps its exact behaviour and message.
    for task in ("t2va", "fl2va"):
        with pytest.raises(ValueError, match=f"ref2va adapter cannot serve task={task}$"):
            _spec().check_request(_sampling(), task)


@pytest.mark.parametrize("overrides", _REJECTED_OVERRIDES)
def test_pika_fl2va_request_cannot_silently_change_sampling(overrides):
    with pytest.raises(ValueError):
        _spec("fl2va").check_request(_sampling(**overrides), "t2va")


@pytest.mark.parametrize("overrides", _ACCEPTED_OVERRIDES)
def test_pika_fl2va_request_accepts_its_own_contract(overrides):
    _spec("fl2va").check_request(_sampling(**overrides), "t2va")


def test_pika_fl2va_pipeline_request_validation():
    pipe = object.__new__(pipeline_module.MiniMaxH3Pipeline)
    pipe.od_config = OmniDiffusionConfig()
    pipe._fasth3 = None
    pipe._pika_spec = _spec("fl2va")
    positions, steps = pipe._resolve_sigma_positions("t2va", _sampling(num_inference_steps=50))
    assert steps == 6 and positions == (1.0, 0.75, 0.5, 0.25, 0.1, 0.04, 0.0)
    assert pipe._validate_pika_request(_sampling(), "t2va") is pipe._pika_spec
    with pytest.raises(OmniClientError, match="not ported"):
        pipe._validate_pika_request(_sampling(), "fl2va")
    with pytest.raises(OmniClientError, match="num_inference_steps"):
        pipe._validate_pika_request(_sampling(num_inference_steps=50), "t2va")


def test_pika_fl2va_startup_claims_the_fl2va_partition_only(tmp_path):
    _artifact(tmp_path, task="fl2va")
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        pipe = _startup_pipe("fl2va")
        spec = pipe._resolve_startup_pika(SimpleNamespace(lora_path=str(tmp_path), lora_scale=1.0), modular=False)
        assert spec is not None and spec.task == "fl2va" and spec.release_id == "pika-flex-fl2va-v1"
        assert spec.numerics_policy == "bf16_conditioning_v1" and spec.conditioning_dtype == torch.bfloat16
        assert spec.request_tasks == frozenset({"t2va"})
        assert pipe.diffusion_lora_fixed_adapter is True
        assert torch.backends.cuda.matmul.allow_tf32 is False
        for partition in ("ref2va", "combined"):
            with pytest.raises(ValueError, match="task-type fl2va"):
                _startup_pipe(partition)._resolve_startup_pika(SimpleNamespace(lora_path=str(tmp_path)), modular=False)
        with pytest.raises(ValueError, match="original"):
            _startup_pipe("fl2va")._resolve_startup_pika(SimpleNamespace(lora_path=str(tmp_path)), modular=True)
        with pytest.raises(ValueError, match="original"):
            _startup_pipe("fl2va", base_schedule=object())._resolve_startup_pika(
                SimpleNamespace(lora_path=str(tmp_path)), modular=False
            )
        with pytest.raises(ValueError, match="cache-backend none"):
            _startup_pipe("fl2va")._resolve_startup_pika(
                SimpleNamespace(lora_path=str(tmp_path), cache_backend="cache_dit"), modular=False
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def test_pika_fl2va_adapter_loads_only_on_the_fl2va_partition(tmp_path, monkeypatch):
    monkeypatch.setattr(lora_module, "_TURBO_TARGET_DIMS", _TINY_TARGET_DIMS)
    path = _artifact(tmp_path, task="fl2va")
    with pytest.raises(ValueError, match="dedicated"):
        _load(path, "ref2va")
    pipe = _startup_pipe("fl2va")
    pipe._turbo_lora_specs, pipe._native_distilled_lora_adapter_ids, pipe._lora_sigma_schedules = {}, set(), {}
    pipe._pika_lora_specs = {}
    pipe._pika_spec = read_pika_spec(path)
    request = LoRARequest("pika", 7, str(tmp_path))
    model, _ = pipe._load_diffusion_lora_adapter(lora_request=request, lora_path=str(tmp_path), dtype=torch.bfloat16)
    assert pipe._pika_lora_specs[7] == pipe._pika_spec and pipe._pika_spec.task == "fl2va"
    pipe._validate_diffusion_lora_binding(lora_model=model, bound_lora_names=frozenset(model.loras))


def test_pika_t2va_request_preparation_and_six_step_renoise_loop():
    """t2va under the fl2va adapter: no visual anchor, Pika's device noise, ladders and re-noise loop."""
    model = _RecordingModel()
    pipe = _step_pipeline(model)
    spec = _spec("fl2va")
    inputs = pipe._build_denoise_inputs(
        task="t2va",
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
        visual_condition=None,
        visual_condition_shape=None,
        audio_condition=None,
        ref_audio_t=None,
        pika=spec,
        # The base runtime can prepare CPU noise ahead of encoding; Pika ignores it.
        precomputed_initial_noise=(torch.zeros(12, 96), torch.zeros(6, 32)),
    )
    branch = inputs["branch"]
    assert inputs["cond_anchor"] is None and inputs["audio_anchor"] is None
    assert bool(branch.update_mask.all()) and bool(branch.audio_update_mask.all())
    initial_video, initial_audio = _noise().initial_rows()
    torch.testing.assert_close(inputs["video_rows"], initial_video, rtol=0, atol=0)
    torch.testing.assert_close(inputs["audio_rows"], initial_audio, rtol=0, atol=0)
    assert inputs["sigmas_video"] == pika_sigmas(12.0) and inputs["sigmas_audio"] == pika_sigmas(3.0)
    assert inputs["sampler"] == RENOISE_SAMPLER and isinstance(inputs["noise_source"], PikaNoise)
    out_video, out_audio = minimax_h3_denoise_loop(
        model=model,
        positive=branch,
        initial_video_rows=inputs["video_rows"],
        initial_audio_rows=inputs["audio_rows"],
        keyframe_cond_rows=inputs["cond_anchor"],
        audio_ref_rows=inputs["audio_anchor"],
        sigmas_video=inputs["sigmas_video"],
        sigmas_audio=inputs["sigmas_audio"],
        device=torch.device("cpu"),
        sampler=inputs["sampler"],
        noise_source=inputs["noise_source"],
    )
    assert len(model.calls) == 6
    reference = _noise()
    sv, sa = pika_sigmas(12.0), pika_sigmas(3.0)
    rows_v, rows_a = initial_video.clone(), initial_audio.clone()
    for i, call in enumerate(model.calls):
        torch.testing.assert_close(call["video_rows"], rows_v, rtol=0, atol=0)
        torch.testing.assert_close(call["audio_rows"], rows_a, rtol=0, atol=0)
        x0_v = rows_v + sv[i] * call["video_out"].float()
        x0_a = rows_a + sa[i] * call["audio_out"].float()
        eps = reference.draw(i, (x0_v if sv[i + 1] else None, x0_a if sa[i + 1] else None))
        rows_v = renoise_step(x0_v, eps[0], sigma_next=sv[i + 1])
        rows_a = renoise_step(x0_a, eps[1], sigma_next=sa[i + 1])
    torch.testing.assert_close(out_video, rows_v, rtol=0, atol=0)
    torch.testing.assert_close(out_audio, rows_a, rtol=0, atol=0)

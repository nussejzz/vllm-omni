# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for in-place decode revert and stage-boundary cache release.

``_revert_decoded_inplace`` must be numerically identical to the checkpoint's
``revert_tensor`` (denormalize → clamp → rearrange) with zero whole-video
copies, and ``_release_stage_cache`` must force the DLO component cache's
release. Without one, a fully resident pipeline on a discrete device releases
only past the bounded policy's bounds; every other setup (the decoder-only
stage, any offload strategy, HSDP, an integrated device, or
``VLLM_OMNI_H3_STAGE_CACHE_RELEASE=force``) releases at every boundary.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 as pipeline_module
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3_decoder import MiniMaxH3DecoderPipeline
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE
from vllm_omni.diffusion.offloader import BoundedAllocatorCache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

DENORM_MEAN = (-2.11628412, -2.03571429, -1.80444444)  # -imagenet_mean/std
DENORM_STD = (4.36681223, 4.46428571, 4.44444444)  # 1/imagenet_std


class _FakeDenormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std


class _FakeProcessor:
    def __init__(self):
        self.transform_rev = _FakeDenormalize(DENORM_MEAN, DENORM_STD)
        self.use_3d_conv = True
        self.revert_calls = 0

    def revert_tensor(self, tensor):
        self.revert_calls += 1
        return _legacy_revert(tensor)


class _FakeCheckpointModel:
    def __init__(self):
        self.processor = _FakeProcessor()


def _vae(model=None):
    vae = object.__new__(MiniMaxH3VideoVAE)
    vae.model = model if model is not None else _FakeCheckpointModel()
    return vae


def _legacy_revert(tensor: torch.Tensor) -> torch.Tensor:
    """The checkpoint's revert_tensor, op for op, on the CPU (B == 1)."""
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(2)
    assert tensor.ndim == 5
    # Normalize runs on the (b t) c h w rearrangement; at B == 1 that is a
    # view of (B, C, T, H, W), so the per-channel broadcast is equivalent.
    mean = torch.tensor(DENORM_MEAN).view(1, 3, 1, 1, 1)
    std = torch.tensor(DENORM_STD).view(1, 3, 1, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.clamp(0, 1)


def test_revert_inplace_matches_legacy_bitwise():
    decoded = torch.randn(1, 3, 7, 6, 8).clamp_(-3, 3)
    expected = _legacy_revert(decoded.clone())
    out = _vae()._revert_decoded_inplace(decoded.clone())
    assert out.shape == expected.shape == (1, 3, 7, 6, 8)
    assert torch.equal(out, expected)


def test_revert_inplace_mutates_without_extra_copies():
    decoded = torch.randn(1, 3, 5, 4, 4)
    out = _vae()._revert_decoded_inplace(decoded)
    assert out is decoded  # in-place: no whole-video copy materializes
    assert out.is_contiguous()


def test_revert_inplace_unsqueezes_4d_input():
    # 4-D decoded tensors follow the (B, C, H, W) image contract: unsqueeze(2)
    # lands T=1, matching the checkpoint's revert_tensor branch.
    decoded = torch.randn(1, 3, 6, 8)
    expected = _legacy_revert(decoded.clone())
    out = _vae()._revert_decoded_inplace(decoded.clone())
    assert torch.equal(out, expected)


def test_revert_inplace_falls_back_without_contract():
    vae = _vae()
    vae.model.processor.transform_rev = type("Empty", (), {})()
    decoded = torch.randn(1, 3, 5, 4, 4)
    out = vae._revert_decoded_inplace(decoded)
    assert vae.model.processor.revert_calls == 1
    assert torch.equal(out, _legacy_revert(decoded))


def test_revert_inplace_falls_back_on_bad_arity():
    vae = _vae()
    vae.model.processor.transform_rev = _FakeDenormalize((0.5, 0.5), (1.0, 1.0, 1.0))
    vae._revert_decoded_inplace(torch.randn(1, 3, 5, 4, 4))
    assert vae.model.processor.revert_calls == 1


class _RecordingCache:
    def __init__(self):
        self.calls = []

    def release_if_needed(self, *, force=False):
        self.calls.append(force)
        return True


_RELEASE_ENV = pipeline_module.MINIMAX_H3_STAGE_CACHE_RELEASE_ENV
_GIB = 1 << 30
# The 8xH200 USP8 decode boundary, scaled to the 100 GiB test device: ~4% of
# capacity idle, ~40% physically free.
_WITHIN_BOUNDS = (58, 54, 40)


class _Allocator:
    """Device telemetry the bounded policy reads, on a 100 GiB device."""

    def __init__(self, monkeypatch):
        self.reserved = self.allocated = 0
        self.free = 100 * _GIB
        self.queried = []
        self.released = 0
        platform = pipeline_module.current_omni_platform
        monkeypatch.setattr(platform, "empty_cache", self._empty_cache, raising=False)
        monkeypatch.setattr(platform, "get_device_memory", self._device_memory)
        monkeypatch.setattr(torch.accelerator, "memory_reserved", lambda _device: self.reserved)
        monkeypatch.setattr(torch.accelerator, "memory_allocated", lambda _device: self.allocated)

    def set_gib(self, reserved, allocated, free):
        self.reserved, self.allocated, self.free = reserved * _GIB, allocated * _GIB, free * _GIB

    def _empty_cache(self):
        self.released += 1

    def _device_memory(self, device):
        self.queried.append(device)
        return self.free, 100 * _GIB


@pytest.fixture
def allocator(monkeypatch):
    monkeypatch.delenv(_RELEASE_ENV, raising=False)
    return _Allocator(monkeypatch)


def _pipeline(device="cpu", **od_config):
    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline._dlo_component_cache = None
    pipeline.od_config = SimpleNamespace(**od_config)
    pipeline.device = torch.device(device)
    return pipeline


def _decoder_pipeline():
    # Mirrors MiniMaxH3DecoderPipeline.__init__, which sets no DLO component cache.
    pipeline = object.__new__(MiniMaxH3DecoderPipeline)
    pipeline.od_config = SimpleNamespace()
    pipeline.device = torch.device("cpu")
    return pipeline


def _no_config_pipeline():
    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline._dlo_component_cache = None
    pipeline.device = torch.device("cpu")
    return pipeline


@pytest.mark.parametrize("mode", [None, "bounded", "force"])
def test_stage_cache_release_forces_component_cache(monkeypatch, allocator, mode):
    if mode is not None:
        monkeypatch.setenv(_RELEASE_ENV, mode)
    pipeline = _pipeline()
    cache = _RecordingCache()
    pipeline._dlo_component_cache = cache
    pipeline._release_stage_cache()
    assert cache.calls == [True]
    assert allocator.released == 0
    assert not allocator.queried


@pytest.mark.parametrize("mode", [None, "", "bounded", " bounded "])
def test_stage_cache_release_keeps_idle_pages_within_bounds(monkeypatch, allocator, mode):
    if mode is not None:
        monkeypatch.setenv(_RELEASE_ENV, mode)
    allocator.set_gib(*_WITHIN_BOUNDS)
    pipeline = _pipeline()
    pipeline._release_stage_cache()
    cache = pipeline._stage_allocator_cache
    pipeline._release_stage_cache()
    assert allocator.released == 0
    assert set(allocator.queried) == {pipeline.device}
    assert isinstance(cache, BoundedAllocatorCache) and cache.device == pipeline.device
    assert pipeline._stage_allocator_cache is cache


def test_stage_cache_release_bounds_are_inclusive(allocator):
    allocator.set_gib(79, 54, 5)  # exactly 25% of capacity idle and 5% free
    _pipeline()._release_stage_cache()
    assert allocator.released == 0


@pytest.mark.parametrize(
    ("reserved", "allocated", "free"),
    [
        (60, 34, 40),  # 26% of capacity idle
        (60, 55, 4),  # 4% of capacity physically free
    ],
)
def test_stage_cache_release_releases_past_bounds(allocator, reserved, allocated, free):
    allocator.set_gib(reserved, allocated, free)
    _pipeline()._release_stage_cache()
    assert allocator.released == 1


def test_stage_cache_release_without_telemetry_releases(monkeypatch, allocator):
    def unavailable(_device):
        raise RuntimeError("no device memory telemetry")

    monkeypatch.setattr(pipeline_module.current_omni_platform, "get_device_memory", unavailable)
    allocator.set_gib(*_WITHIN_BOUNDS)
    _pipeline()._release_stage_cache()
    assert allocator.released == 1


def test_stage_cache_release_force_mode_always_empties_cache(monkeypatch, allocator):
    monkeypatch.setenv(_RELEASE_ENV, "force")
    allocator.set_gib(*_WITHIN_BOUNDS)
    pipeline = _pipeline()
    pipeline._release_stage_cache()
    pipeline._release_stage_cache()
    assert allocator.released == 2
    assert not allocator.queried


@pytest.mark.parametrize("mode", [None, "bounded"])
@pytest.mark.parametrize(
    "make_pipeline",
    [
        pytest.param(_decoder_pipeline, id="decoder-stage"),
        pytest.param(lambda: _pipeline(enable_cpu_offload=True), id="model-offload"),
        pytest.param(lambda: _pipeline(enable_layerwise_offload=True), id="layerwise-offload"),
        pytest.param(lambda: _pipeline(enable_distributed_layerwise_offload=True), id="dlo-without-cache"),
        pytest.param(lambda: _pipeline(parallel_config=SimpleNamespace(use_hsdp=True)), id="hsdp"),
        pytest.param(_no_config_pipeline, id="no-config"),
    ],
)
def test_stage_cache_release_forces_outside_resident_pipelines(monkeypatch, allocator, mode, make_pipeline):
    if mode is not None:
        monkeypatch.setenv(_RELEASE_ENV, mode)
    allocator.set_gib(*_WITHIN_BOUNDS)
    pipeline = make_pipeline()
    pipeline._release_stage_cache()
    pipeline._release_stage_cache()
    assert allocator.released == 2
    assert not allocator.queried
    assert getattr(pipeline, "_stage_allocator_cache", None) is None


@pytest.mark.parametrize(("integrated", "released"), [(True, 2), (False, 0)])
def test_stage_cache_release_forces_on_integrated_devices(monkeypatch, allocator, integrated, released):
    properties = []

    def device_properties(device):
        properties.append(device)
        return SimpleNamespace(is_integrated=integrated)

    monkeypatch.setattr(torch.cuda, "get_device_properties", device_properties)
    allocator.set_gib(*_WITHIN_BOUNDS)
    pipeline = _pipeline("cuda:0")
    pipeline._release_stage_cache()
    pipeline._release_stage_cache()
    assert allocator.released == released
    assert properties and set(properties) == {pipeline.device}


@pytest.mark.parametrize("raw", ["Force", "always", "0", "off"])
@pytest.mark.parametrize("make_pipeline", [_pipeline, _decoder_pipeline])
def test_stage_cache_release_rejects_unknown_mode(monkeypatch, allocator, raw, make_pipeline):
    monkeypatch.setenv(_RELEASE_ENV, raw)
    with pytest.raises(ValueError, match=_RELEASE_ENV):
        pipeline_module._stage_cache_release_mode()
    with pytest.raises(ValueError, match=_RELEASE_ENV):
        make_pipeline()._release_stage_cache()
    assert allocator.released == 0

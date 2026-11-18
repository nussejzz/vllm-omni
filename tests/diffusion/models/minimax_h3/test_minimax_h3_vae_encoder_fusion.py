# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the MiniMax-H3 video VAE encoder fusion installer.

The exact CUDA operator is replaced by a torch reference with the same
contract, so the installed forwards are checked bit for bit against the
checkpoint forwards without an accelerator.
"""

import copy
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from tests.diffusion.models.minimax_h3.h3_vae_encoder_replica import make_encoder
from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module
from vllm_omni.diffusion.models.minimax_h3.ops.vae import encoder as encoder_ops
from vllm_omni.diffusion.models.minimax_h3.ops.vae.dispatch import H3VAEOperatorSet

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def reference_pad(source, *, front, top, bottom, left, right, apply_silu):
    """The checkpoint chain the fused operator replaces, op for op."""

    source = source.contiguous()
    if apply_silu:
        source = F.silu(source)
    padded = F.pad(source, (left, right, top, bottom, 0, 0), mode="reflect")
    return F.pad(padded, (0, 0, 0, 0, front, 0))


def _use_operator(monkeypatch, pad_op):
    calls = []

    def recording(source, **pads):
        calls.append(pads)
        return pad_op(source, **pads)

    operators = H3VAEOperatorSet(
        supports=lambda _device: True,
        qk_norm_rope=lambda *_args: None,
        scaled_residual=lambda *_args: None,
        causal_reflect_pad=recording,
    )
    monkeypatch.setattr(encoder_ops, "resolve_h3_vae_operators", lambda _device: operators)
    return calls


def _inputs():
    torch.manual_seed(3)
    video = torch.randn(1, 3, 40, 48, 40)
    return {
        "video": video[:, :, 3:20, 8:40, 4:36],  # non-contiguous clip tile
        "frame": video[:, :, 7:8, :32, :32],  # single-frame (image) branch
        "batch": torch.randn(2, 3, 5, 32, 16),
    }


@pytest.mark.parametrize("time_isolated", [True, False])
def test_encoder_fusion_matches_checkpoint_forward(monkeypatch, time_isolated):
    calls = _use_operator(monkeypatch, reference_pad)
    reference = make_encoder(time_isolated=time_isolated)
    fused = copy.deepcopy(reference)
    # ``apply_spatial_parallel`` assigns the checkpoint's JSON integer flag.
    for module in fused.modules():
        if hasattr(module, "spatial_parallel"):
            module.spatial_parallel = 0

    assert encoder_ops.install_h3_vae_encoder_fusion(fused, device=torch.device("cpu"))
    assert fused._omni_h3_vae_encoder_fusion_installed
    with torch.inference_mode():
        for name, x in _inputs().items():
            expected = reference(x)
            actual = fused(x)
            assert torch.equal(actual, expected), name
    # conv_in, 6 blocks x 2 norm-SiLU-convs, 2 downsamplers and conv_out,
    # for each of the three inputs.
    assert len(calls) == 3 * (1 + 12 + 2 + 1)
    assert {"front": 2, "top": 0, "bottom": 1, "left": 0, "right": 1, "apply_silu": False} in calls
    assert sum(call["apply_silu"] for call in calls) == 3 * 13


def test_encoder_fusion_falls_back_when_operator_declines(monkeypatch):
    calls = _use_operator(monkeypatch, lambda _source, **_pads: None)
    reference = make_encoder()
    fused = copy.deepcopy(reference)

    assert encoder_ops.install_h3_vae_encoder_fusion(fused, device=torch.device("cpu"))
    with torch.inference_mode():
        for name, x in _inputs().items():
            assert torch.equal(fused(x), reference(x)), name
    assert calls


def test_encoder_fusion_keeps_conditioned_and_autograd_paths(monkeypatch):
    calls = _use_operator(monkeypatch, reference_pad)
    reference = make_encoder()
    fused = copy.deepcopy(reference)
    assert encoder_ops.install_h3_vae_encoder_fusion(fused, device=torch.device("cpu"))

    hidden = torch.randn(1, 32, 3, 8, 8)
    with torch.inference_mode():
        # A conditioning input is outside the contract: the checkpoint block
        # forward runs, and only its convolutions use the fused padding.
        actual = fused.down[0].block[0](hidden, torch.zeros(1))
        assert torch.equal(actual, reference.down[0].block[0](hidden))
    assert calls and not any(call["apply_silu"] for call in calls)

    x = _inputs()["video"]
    actual = fused(x)
    expected = reference(x)
    assert actual.requires_grad and torch.equal(actual, expected)


def test_encoder_fusion_is_installed_once(monkeypatch):
    _use_operator(monkeypatch, reference_pad)
    encoder = make_encoder()
    assert encoder_ops.install_h3_vae_encoder_fusion(encoder, device=torch.device("cpu"))
    forward = encoder.down[0].block[0].forward
    assert encoder_ops.install_h3_vae_encoder_fusion(encoder, device=torch.device("cpu"))
    assert encoder.down[0].block[0].forward is forward


def test_encoder_fusion_requires_registered_operator(monkeypatch):
    monkeypatch.setattr(encoder_ops, "resolve_h3_vae_operators", lambda _device: None)
    encoder = make_encoder()
    assert not encoder_ops.install_h3_vae_encoder_fusion(encoder, device=torch.device("cpu"))
    assert "forward" not in vars(encoder)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda encoder: setattr(encoder, "use_fused_norm", True),
        lambda encoder: setattr(encoder.down[1].block[0], "use_fused_norm", True),
        lambda encoder: setattr(encoder.down[0].block[1].conv1, "pad_mode", "replicate"),
        lambda encoder: setattr(encoder.conv_out, "causal", False),
        lambda encoder: setattr(encoder.norm_out, "spatial_parallel", True),
        lambda encoder: setattr(encoder.down[0].downsample, "pad_mode", "constant"),
        # A remote padding implementation that no longer matches the contract.
        lambda encoder: setattr(encoder.conv_in, "_apply_padding", lambda x: F.pad(x, (1, 1, 1, 1, 1, 1))),
    ],
)
def test_encoder_fusion_rejects_unknown_contracts(monkeypatch, mutate):
    _use_operator(monkeypatch, reference_pad)
    encoder = make_encoder()
    mutate(encoder)
    assert not encoder_ops.install_h3_vae_encoder_fusion(encoder, device=torch.device("cpu"))
    assert not any("forward" in vars(module) for module in encoder.modules())
    assert not getattr(encoder, "_omni_h3_vae_encoder_fusion_installed", False)


def test_encoder_fusion_rejects_zeros_padding(monkeypatch):
    _use_operator(monkeypatch, reference_pad)
    encoder = make_encoder(padding_mode="zeros")
    assert not encoder_ops.install_h3_vae_encoder_fusion(encoder, device=torch.device("cpu"))


@pytest.mark.parametrize(("value", "expected"), [(None, True), ("", True), ("1", True), (" 0 ", False)])
def test_encoder_fusion_env_switch(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("VLLM_OMNI_H3_VAE_ENCODER_FUSION", raising=False)
    else:
        monkeypatch.setenv("VLLM_OMNI_H3_VAE_ENCODER_FUSION", value)
    assert vae_module._encoder_fusion_enabled() is expected


def test_encoder_fusion_env_switch_rejects_unknown_values(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_H3_VAE_ENCODER_FUSION", "true")
    with pytest.raises(ValueError, match="VLLM_OMNI_H3_VAE_ENCODER_FUSION"):
        vae_module._encoder_fusion_enabled()


@pytest.mark.parametrize(("value", "installed"), [(None, True), ("0", False)])
def test_video_vae_installs_encoder_fusion_unless_disabled(monkeypatch, value, installed):
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Module()
            self.decoder = torch.nn.Module()

    class FakeRemote(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeModel()

    monkeypatch.setattr(
        vae_module,
        "_load_component_config",
        lambda _path: {"latent_channels": 1, "latents_mean": [0.0], "latents_std": [1.0]},
    )
    monkeypatch.setattr(vae_module, "_load_remote_component", lambda _path, _config, **_kwargs: FakeRemote())
    monkeypatch.setattr(vae_module, "install_h3_vae_optimizations", Mock(return_value=True))
    install = Mock(return_value=True)
    monkeypatch.setattr(vae_module, "install_h3_vae_encoder_fusion", install)
    if value is None:
        monkeypatch.delenv("VLLM_OMNI_H3_VAE_ENCODER_FUSION", raising=False)
    else:
        monkeypatch.setenv("VLLM_OMNI_H3_VAE_ENCODER_FUSION", value)

    video_vae = vae_module.MiniMaxH3VideoVAE("unused", device=torch.device("cpu"))

    if installed:
        install.assert_called_once_with(video_vae.model.encoder, device=torch.device("cpu"))
    else:
        install.assert_not_called()

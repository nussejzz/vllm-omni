# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bit-exactness of the fused MiniMax-H3 VAE encoder padding operator."""

import copy

import pytest
import torch
import torch.nn.functional as F

from tests.diffusion.models.minimax_h3.h3_vae_encoder_replica import make_encoder
from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.minimax_h3.ops.vae.encoder import install_h3_vae_encoder_fusion
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": ["H100"]}, num_cards=1),
]


def _pad_operator():
    from vllm_omni.diffusion.models.minimax_h3.ops.vae.dispatch import resolve_h3_vae_operators

    if not current_omni_platform.is_available():
        pytest.skip("No accelerator is available")
    device = current_omni_platform.get_torch_device()
    operators = resolve_h3_vae_operators(device)
    if operators is None or operators.causal_reflect_pad is None:
        pytest.skip("No exact H3 VAE encoder operator is registered for this target")
    return device, operators.causal_reflect_pad


def _checkpoint_chain(source, *, front, top, bottom, left, right, apply_silu):
    source = source.contiguous()
    if apply_silu:
        source = F.silu(source)
    padded = F.pad(source, (left, right, top, bottom, 0, 0), mode="reflect")
    if padded.shape[2] > 1:
        return F.pad(padded, (0, 0, 0, 0, front, 0))
    zeros = torch.zeros_like(padded[:, :, :1]).expand(-1, -1, front, -1, -1)
    return torch.cat([zeros, padded], dim=2)


def _assert_same_bits(actual, expected):
    assert actual.shape == expected.shape
    assert actual.is_contiguous()
    assert torch.equal(actual.view(torch.int32), expected.contiguous().view(torch.int32))


@pytest.mark.parametrize(
    "shape",
    [(1, 128, 17, 64, 64), (1, 64, 9, 33, 17), (2, 32, 1, 16, 16), (1, 32, 5, 2, 300)],
)
def test_fused_norm_output_padding_is_bit_exact(shape):
    device, pad_op = _pad_operator()
    batch, channels, frames, height, width = shape
    generator = torch.Generator(device=device).manual_seed(11)
    # GroupNorm output layout: BTCHW memory read as a BCTHW view.
    normalized = 4 * torch.randn(batch, frames, channels, height, width, device=device, generator=generator)
    source = normalized.permute(0, 2, 1, 3, 4)
    pads = {"front": 2, "top": 1, "bottom": 1, "left": 1, "right": 1}

    with torch.inference_mode():
        for apply_silu in (True, False):
            actual = pad_op(source, **pads, apply_silu=apply_silu)
            assert actual is not None
            _assert_same_bits(actual, _checkpoint_chain(source, **pads, apply_silu=apply_silu))


@pytest.mark.parametrize("pads", [(2, 0, 1, 0, 1), (2, 0, 0, 0, 0), (0, 1, 1, 1, 1), (3, 1, 0, 0, 1)])
def test_downsample_and_input_padding_is_bit_exact(pads):
    device, pad_op = _pad_operator()
    front, top, bottom, left, right = pads
    video = torch.randn(1, 3, 40, 96, 80, device=device)
    tile = video[:, :, 5:22, 16:80, 8:72]  # non-contiguous clip tile

    with torch.inference_mode():
        for source in (tile, tile.contiguous(), tile[:, :, :1]):
            kwargs = {"front": front, "top": top, "bottom": bottom, "left": left, "right": right, "apply_silu": False}
            actual = pad_op(source, **kwargs)
            assert actual is not None
            _assert_same_bits(actual, _checkpoint_chain(source, **kwargs))


def test_fused_silu_matches_eager_for_every_fp32_value():
    device, pad_op = _pad_operator()
    chunk = 1 << 26
    with torch.inference_mode():
        for start in range(-(1 << 31), 1 << 31, chunk):
            bits = torch.arange(start, start + chunk, dtype=torch.int64, device=device).to(torch.int32)
            values = bits.view(torch.float32).view(1, 1, 1, chunk // 1024, 1024)
            actual = pad_op(values, front=0, top=0, bottom=0, left=0, right=0, apply_silu=True)
            expected = F.silu(values)
            assert torch.equal(actual.view(torch.int32), expected.view(torch.int32)), hex(start)


def test_fused_padding_rejects_inputs_outside_its_contract():
    device, pad_op = _pad_operator()
    source = torch.randn(1, 2, 3, 4, 5, device=device)
    pads = {"front": 2, "top": 1, "bottom": 1, "left": 1, "right": 1, "apply_silu": True}

    with torch.inference_mode():
        assert pad_op(source.half(), **pads) is None
        assert pad_op(source.cpu(), **pads) is None
        assert pad_op(source[:, :, :, :1], **pads) is None  # reflect pad >= height
        assert pad_op(source, **{**pads, "left": -1}) is None
    with torch.enable_grad():
        assert pad_op(source.requires_grad_(), **pads) is None


@pytest.mark.parametrize("time_isolated", [True, False])
def test_fused_encoder_matches_checkpoint_forward(time_isolated):
    device, pad_op = _pad_operator()
    reference = make_encoder(time_isolated=time_isolated).to(device)
    fused = copy.deepcopy(reference)
    assert install_h3_vae_encoder_fusion(fused, device=device)
    assert fused.conv_in._omni_causal_reflect_pad is pad_op

    video = torch.randn(1, 3, 40, 80, 72, device=device)
    with torch.inference_mode():
        for x in (video[:, :, 3:20, 8:72, 4:68], video[:, :, 7:8, :64, :64], video[:, :, :5, :32, :48]):
            assert torch.equal(fused(x), reference(x))

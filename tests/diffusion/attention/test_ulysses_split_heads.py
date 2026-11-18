# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Head-split Q/K/V inputs and the token-major output send of strict Ulysses."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel import ulysses
from vllm_omni.diffusion.distributed import a2a_permute

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _sp(world_size: int = 2):
    return SimpleNamespace(
        ulysses_group=SimpleNamespace(group_name="subgroup"),
        ulysses_world_size=world_size,
        ring_world_size=1,
        allgather_world_size=1,
    )


@pytest.mark.parametrize(
    ("world_size", "scatter", "gather", "ce", "mode", "expected"),
    [
        (2, 2, 1, False, "strict", 2),
        (8, 2, 1, False, "strict", 8),
        (1, 2, 1, False, "strict", 0),
        (2, 1, 2, False, "strict", 0),
        (2, 2, 1, True, "strict", 0),
        (2, 2, 1, False, "advanced_uaa", 0),
    ],
)
def test_split_heads_send_degree_is_strict_nccl_only(monkeypatch, world_size, scatter, gather, ce, mode, expected):
    monkeypatch.setattr(a2a_permute, "ensure_a2a_permute_available", lambda: None)
    monkeypatch.setattr(ulysses, "get_ulysses_mode", lambda default: mode)
    strategy = ulysses.UlyssesParallelAttention(_sp(world_size), scatter, gather, False, ulysses_a2a_permute=ce)
    assert strategy.split_heads_send_degree() == expected


def _recording_strategy(monkeypatch, *, ce: bool):
    calls: list[tuple] = []
    monkeypatch.setattr(a2a_permute, "ensure_a2a_permute_available", lambda: None)
    monkeypatch.setattr(ulysses, "get_ulysses_mode", lambda default: "strict")
    monkeypatch.setattr(ulysses.dist, "get_world_size", lambda pg: 2)

    def nccl_exchange(pg, x, scatter, gather, sync, *token_major):
        calls.append(("nccl", tuple(x.shape), scatter, token_major))
        if scatter == 2:
            b, s = x.shape[:2]
            return x.new_zeros(b, s * 2, 4, 64)
        b, s, h, d = x.shape
        return x.new_zeros(b, s // 2, h * 2, d)

    def ce_exchange(q, k, v, group_name, world_size):
        calls.extend(("ce", tuple(t.shape)) for t in (q, k, v))
        return tuple(t.new_zeros(1, 32, 4, 64) for t in (q, k, v))

    monkeypatch.setattr(ulysses.SeqAllToAll4D, "apply", nccl_exchange)
    monkeypatch.setattr(a2a_permute, "ulysses_qkv_fwd_group", ce_exchange)
    monkeypatch.setattr(a2a_permute, "ulysses_o_rev", lambda x, g, w: x.new_zeros(1, 16, 8, 64))
    return ulysses.UlyssesParallelAttention(_sp(), 2, 1, False, ulysses_a2a_permute=ce), calls


def _split_qkv():
    # [P, S/P, H/P, D] send buffers viewed as head-split [1, S/P, P, H/P, D].
    send = torch.randn(2, 16, 4, 64)
    return send.permute(1, 0, 2, 3).unsqueeze(0), send


def test_strict_nccl_keeps_head_split_inputs_and_token_major_output(monkeypatch):
    strategy, calls = _recording_strategy(monkeypatch, ce=False)
    split, _ = _split_qkv()
    metadata = AttentionMetadata(extra={ulysses.ULYSSES_TOKEN_MAJOR_OUTPUT_KEY: True})
    q, _, _, _, ctx = strategy.pre_attention(split, split, split, metadata)
    assert ctx.token_major_output
    assert strategy.post_attention(q, ctx).shape == (1, 16, 8, 64)
    split_call = ("nccl", (1, 16, 2, 4, 64), 2, ())
    assert calls == [split_call] * 3 + [("nccl", (1, 32, 4, 64), 1, (True,))]


def test_plain_inputs_keep_the_permuting_output_send(monkeypatch):
    strategy, calls = _recording_strategy(monkeypatch, ce=False)
    q = torch.randn(1, 16, 8, 64)
    q, _, _, _, ctx = strategy.pre_attention(q, q, q, AttentionMetadata())
    assert not ctx.token_major_output
    strategy.post_attention(q, ctx)
    assert calls[-1] == ("nccl", (1, 32, 4, 64), 1, ())


def test_copy_engine_transport_merges_head_split_inputs(monkeypatch):
    strategy, calls = _recording_strategy(monkeypatch, ce=True)
    split, send = _split_qkv()
    metadata = AttentionMetadata(extra={ulysses.ULYSSES_TOKEN_MAJOR_OUTPUT_KEY: True})
    q, _, _, _, ctx = strategy.pre_attention(split, split, split, metadata)
    assert calls == [("ce", (1, 16, 8, 64))] * 3
    # The copy-engine output path never sees the NCCL-only flag.
    assert strategy.post_attention(q, ctx).shape == (1, 16, 8, 64)
    assert len(calls) == 3


def test_vsa_gate_merges_head_split_inputs(monkeypatch):
    strategy, calls = _recording_strategy(monkeypatch, ce=False)
    split, send = _split_qkv()
    gate = torch.randn(1, 16, 8, 64)
    strategy.pre_attention(split, split, split, AttentionMetadata(extra={"gate_compress": gate}))
    assert [shape for _, shape, *_ in calls] == [(1, 16, 8, 64)] * 4
    merged = split.flatten(2, 3)
    assert torch.equal(merged, send.permute(1, 0, 2, 3).reshape(16, 8, 64).unsqueeze(0))

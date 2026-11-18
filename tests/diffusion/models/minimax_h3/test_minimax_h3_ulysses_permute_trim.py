# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax-H3 strict-Ulysses permute trim: switch, gating and attention metadata."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.parallel import ulysses
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope_head_groups
from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_blocks as blocks

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize(("raw", "expected"), [(None, True), ("", True), ("1", True), (" 0 ", False)])
def test_switch_defaults_on_and_zero_restores_permute_copies(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(blocks.MINIMAX_H3_ULYSSES_PERMUTE_TRIM_ENV, raising=False)
    else:
        monkeypatch.setenv(blocks.MINIMAX_H3_ULYSSES_PERMUTE_TRIM_ENV, raw)
    assert blocks._ulysses_permute_trim_enabled() is expected


def test_switch_rejects_other_values(monkeypatch):
    monkeypatch.setenv(blocks.MINIMAX_H3_ULYSSES_PERMUTE_TRIM_ENV, "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        blocks._ulysses_permute_trim_enabled()


class _Layer:
    def __init__(self, strategy, *, use_ring=False):
        self.strategy = strategy
        self.use_ring = use_ring
        self.skip_sequence_parallel = False

    def _get_active_parallel_strategy(self):
        return self.strategy


def _ulysses(monkeypatch, *, ce=False, world_size=8):
    monkeypatch.setattr(ulysses, "get_ulysses_mode", lambda default: "strict")
    sp = SimpleNamespace(ulysses_group=None, ulysses_world_size=world_size, ring_world_size=1, allgather_world_size=1)
    strategy = ulysses.UlyssesParallelAttention(sp, 2, 1, False)
    # Flip the transport after construction so the copy-engine backend is never initialized.
    strategy._ulysses_a2a_permute = ce
    return strategy


def test_send_head_groups_follow_the_active_strict_nccl_strategy(monkeypatch):
    strategy = _ulysses(monkeypatch)
    assert blocks._ulysses_send_head_groups(_Layer(strategy), 56) == 8
    # Heads that do not split evenly, the copy-engine transport, Ring, other
    # strategies (including Ulysses subclasses) and layers without SP keep Q/K plain.
    assert blocks._ulysses_send_head_groups(_Layer(strategy), 12) == 0
    assert blocks._ulysses_send_head_groups(_Layer(_ulysses(monkeypatch, ce=True)), 56) == 0
    assert blocks._ulysses_send_head_groups(_Layer(strategy, use_ring=True), 56) == 0
    assert blocks._ulysses_send_head_groups(_Layer(NoParallelAttention()), 56) == 0
    subclass = type("ComposedUlysses", (ulysses.UlyssesParallelAttention,), {})
    composed = object.__new__(subclass)
    composed.__dict__.update(strategy.__dict__)
    assert blocks._ulysses_send_head_groups(_Layer(composed), 56) == 0
    assert blocks._ulysses_send_head_groups(SimpleNamespace(), 56) == 0


class _FakeBackend:
    supports_prefix_kv_slicing = True

    @classmethod
    def supports_packed_mask_free(cls) -> bool:
        return False


class _RecordingAttention(torch.nn.Module):
    attn_backend = _FakeBackend

    def __init__(self):
        super().__init__()
        self.calls = []

    def effective_attention(self):
        return self, self.attn_backend, None

    def forward(self, query, key, value, metadata):
        self.calls.append((query.shape, metadata))
        return query.flatten(2, 3) if query.dim() == 5 else query


@pytest.mark.parametrize("token_major", [False, True])
def test_packed_attention_forwards_head_split_qk_and_output_flag(token_major):
    attention = object.__new__(blocks.MiniMaxH3Attention)
    torch.nn.Module.__init__(attention)
    attention.attention = _RecordingAttention()
    split = torch.randn(4, 2, 3, 8)
    out = attention._run_packed_attention(
        split,
        split,
        torch.randn(4, 6, 8),
        cu_seqlens=torch.tensor([0, 8], dtype=torch.int32),
        max_seqlen=8,
        packed_total=8,
        ulysses_token_major_output=token_major,
    )
    assert out.shape == (4, 6, 8)
    shape, metadata = attention.attention.calls[0]
    assert shape == (1, 4, 2, 3, 8)
    assert metadata.extra.get(ulysses.ULYSSES_TOKEN_MAJOR_OUTPUT_KEY, False) is token_major


def test_head_group_rope_defers_to_plain_path_off_cuda():
    q = torch.randn(4, 8, 128, dtype=torch.bfloat16)
    weight = torch.ones(128, dtype=torch.bfloat16)
    table = torch.zeros(4, 96, dtype=torch.bfloat16)
    assert fused_qk_norm_rope_head_groups(q, q, weight, weight, table, 1e-6, 2) is None
    with pytest.raises(ValueError, match="divide"):
        fused_qk_norm_rope_head_groups(q, q, weight, weight, table, 1e-6, 3)


def test_cpu_forward_never_requests_the_trim(monkeypatch):
    calls = []
    monkeypatch.setattr(blocks, "_ULYSSES_PERMUTE_TRIM", True)
    monkeypatch.setattr(blocks, "_ulysses_send_head_groups", lambda *args: calls.append(args) or 8)
    attention = object.__new__(blocks.MiniMaxH3Attention)
    torch.nn.Module.__init__(attention)
    attention.to_gate_compress = attention.vdn = None
    attention.num_heads, attention.head_dim = 2, 4
    attention.attention = SimpleNamespace()
    seen = {}

    def project(x, rope_table, vdn_window, send_head_groups):
        seen["groups"] = send_head_groups
        q = torch.zeros(x.shape[0], 2, 4)
        return q, q, q, q, q

    attention._project_qkv = project
    attention._run_packed_attention = lambda *args, **kwargs: seen.update(kwargs) or torch.zeros(3, 2, 4)
    attention.out_proj = lambda out: (out[:, :1], None)
    attention.forward(torch.zeros(3, 8), rope_table=torch.zeros(3, 96), cu_seqlens=None, max_seqlen=3)
    assert seen["groups"] == 0 and seen["ulysses_token_major_output"] is False
    assert calls == []

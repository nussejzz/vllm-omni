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


@pytest.mark.parametrize(
    ("trim", "rope", "gate", "vdn", "expected"),
    [
        (True, True, False, False, 8),
        # The switch at 0, no RoPE table, a VSA gate or a VDN branch keep the permute copies.
        (False, True, False, False, 0),
        (True, False, False, False, 0),
        (True, True, True, False, 0),
        (True, True, False, True, 0),
    ],
)
def test_cuda_forward_requests_the_trim_on_the_plain_dense_path_only(monkeypatch, trim, rope, gate, vdn, expected):
    monkeypatch.setattr(blocks, "_ULYSSES_PERMUTE_TRIM", trim)
    monkeypatch.setattr(blocks, "_ulysses_send_head_groups", lambda layer, heads: 8)
    attention = object.__new__(blocks.MiniMaxH3Attention)
    torch.nn.Module.__init__(attention)
    attention.num_heads, attention.head_dim = 2, 4
    attention.attention = SimpleNamespace()
    attention.to_gate_compress = (lambda x: torch.zeros(3, 8)) if gate else None
    attention.vdn = SimpleNamespace(gate_softmax=lambda out, x: out, add_linear=lambda out, *a: out) if vdn else None
    seen = {}

    def project(x, rope_table, vdn_window, send_head_groups):
        seen["groups"] = send_head_groups
        q = torch.zeros(x.shape[0], 2, 4)
        return q, q, q, q, q

    attention._project_qkv = project
    attention._run_packed_attention = lambda *args, **kwargs: seen.update(kwargs) or torch.zeros(3, 2, 4)
    attention.out_proj = lambda out: (out[:, :1], None)
    # Only shape and device are read before the (stubbed) projection.
    x = SimpleNamespace(shape=(3, 8), is_cuda=True)
    attention.forward(x, rope_table=torch.zeros(3, 96) if rope else None, cu_seqlens=None, max_seqlen=3)
    assert seen["groups"] == expected
    assert seen["ulysses_token_major_output"] is (expected > 0)


def _projecting_attention(heads=4, head_dim=8):
    attention = object.__new__(blocks.MiniMaxH3Attention)
    torch.nn.Module.__init__(attention)
    attention.num_heads = attention.num_kv_heads = heads
    attention.head_dim = head_dim
    attention.vdn = None
    attention.qkv_proj = lambda x: (x.repeat(1, 3), None)
    norm = SimpleNamespace(weight=torch.ones(head_dim), variance_epsilon=1e-6)
    attention.q_norm = attention.k_norm = norm
    return attention


@pytest.mark.parametrize("kernel_runs", [True, False])
def test_requested_trim_logs_its_outcome_once_per_process(monkeypatch, kernel_runs):
    # vllm's logger does not propagate to caplog; record its calls directly.
    logged = []
    monkeypatch.setattr(blocks.logger, "info", lambda message, *args: logged.append(("info", message % args)))
    monkeypatch.setattr(blocks.logger, "warning", lambda message, *args: logged.append(("warning", message % args)))
    monkeypatch.setattr(blocks, "_PERMUTE_TRIM_REPORTED", set())

    def head_groups(q, k, q_weight, k_weight, table, eps, groups):
        if not kernel_runs:
            return None
        return tuple(t.unflatten(1, (groups, -1)).permute(1, 0, 2, 3).contiguous() for t in (q, k))

    monkeypatch.setattr(blocks, "fused_qk_norm_rope_head_groups", head_groups)
    monkeypatch.setattr(blocks, "fused_qk_norm_rope", lambda q, k, *args: (q, k))
    attention = _projecting_attention()
    x = torch.randn(3, 32)
    for _ in range(3):
        q, k, *_ = attention._project_qkv(x, torch.zeros(3, 4), None, 2)
        assert q.shape == ((3, 2, 2, 8) if kernel_runs else (3, 4, 8))
    # No trim requested: nothing is logged.
    attention._project_qkv(x, torch.zeros(3, 4), None, 0)
    if kernel_runs:
        expected = [
            ("info", "MiniMax-H3 strict-Ulysses permute trim active: Q/K written in the 2-way all-to-all send layout")
        ]
    else:
        expected = [
            (
                "warning",
                "MiniMax-H3 strict-Ulysses permute trim requested (2-way) but the head-group Q/K kernel "
                "does not run for this input; it keeps the Q/K permute copies",
            )
        ]
    assert logged == expected

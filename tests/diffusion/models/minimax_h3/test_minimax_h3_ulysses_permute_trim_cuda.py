# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax-H3 Q/K projection into the strict-Ulysses send layout is bit-exact."""

from types import SimpleNamespace

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_blocks as blocks

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required")
def test_project_qkv_send_layout_matches_plain_projection():
    tokens, heads, head_dim, groups = 333, 56, 128, 8
    torch.manual_seed(31)
    qkv = torch.randn(tokens, 3 * heads * head_dim, device="cuda", dtype=torch.bfloat16)
    attention = object.__new__(blocks.MiniMaxH3Attention)
    torch.nn.Module.__init__(attention)
    attention.qkv_proj = lambda x: (qkv, None)
    attention.num_heads = attention.num_kv_heads = heads
    attention.head_dim = head_dim
    attention.vdn = None
    for name in ("q_norm", "k_norm"):
        weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
        setattr(attention, name, SimpleNamespace(weight=weight, variance_epsilon=1e-6))
    freqs = torch.randn(tokens, 48, device="cuda")
    rope_table = torch.cat((torch.cos(freqs), torch.sin(freqs)), dim=-1).to(torch.bfloat16)
    x = torch.empty(tokens, 1, device="cuda")

    plain = attention._project_qkv(x, rope_table)
    split = attention._project_qkv(x, rope_table, None, groups)
    for index in (0, 1):
        assert split[index].shape == (tokens, groups, heads // groups, head_dim)
        # The head-split view sits on a contiguous [P, T, H/P, D] send buffer.
        assert split[index].permute(1, 0, 2, 3).is_contiguous()
        assert torch.equal(split[index].reshape(tokens, heads, head_dim), plain[index])
    for index in (2, 3, 4):
        assert torch.equal(split[index], plain[index])

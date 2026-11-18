# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""all_to_all_4D layouts that skip a permute copy must exchange the same bytes.

Head-split ``[B, S/P, P, H/P, D]`` inputs written in the send layout and the
batch-1 token-major reverse send are pure layout changes, so their results are
compared with ``torch.equal`` against the permuting path on real gloo groups.
"""

from __future__ import annotations

import os

import pytest
import torch

from tests.helpers.distributed import configure_cpu_collective_worker, start_collective_workers
from tests.helpers.runtime import get_open_port
from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D, all_to_all_4D
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _run_send_layouts(local_rank: int, world_size: int, master_port: int) -> None:
    os.environ.update(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
        }
    )
    configure_cpu_collective_worker()
    init_distributed_environment(backend="gloo")
    try:
        initialize_model_parallel(ulysses_degree=world_size)
        group = get_sp_group().ulysses_group
        generator = torch.Generator().manual_seed(1000 + local_rank)
        tokens, heads, head_dim = 6, 4 * world_size, 8

        x = torch.randn(1, tokens, heads, head_dim, generator=generator)
        expected = all_to_all_4D(x, 2, 1, group=group)
        # A producer writing the [P, S/P, 1, H/P, D] send buffer hands over its
        # [1, S/P, P, H/P, D] view; the exchange then reads it without a copy.
        send = x.reshape(1, tokens, world_size, heads // world_size, head_dim).transpose(0, 2).contiguous()
        split = send.transpose(0, 2)
        assert split.transpose(0, 2).contiguous().data_ptr() == send.data_ptr()
        assert torch.equal(all_to_all_4D(split, 2, 1, group=group), expected)
        assert torch.equal(SeqAllToAll4D.apply(group, split, 2, 1, False), expected)
        # A head-split view that is not in the send layout is permuted as before.
        assert torch.equal(all_to_all_4D(x.reshape(split.shape), 2, 1, group=group), expected)

        out = torch.randn(1, tokens * world_size, heads // world_size, head_dim, generator=generator)
        expected_out = all_to_all_4D(out, 1, 2, group=group)
        assert expected_out.shape == (1, tokens, heads, head_dim)
        assert torch.equal(all_to_all_4D(out, 1, 2, group=group, token_major_send=True), expected_out)
        assert torch.equal(SeqAllToAll4D.apply(group, out, 1, 2, False, True), expected_out)

        # Batch > 1 rows are not contiguous per destination; the flag is ignored.
        batched = torch.randn(2, tokens * world_size, heads // world_size, head_dim, generator=generator)
        assert torch.equal(
            all_to_all_4D(batched, 1, 2, group=group, token_major_send=True),
            all_to_all_4D(batched, 1, 2, group=group),
        )

        with pytest.raises(ValueError, match="rank groups"):
            all_to_all_4D(x.reshape(1, tokens, 2 * world_size, heads // (2 * world_size), head_dim), 2, 1, group=group)
        with pytest.raises(AssertionError):
            all_to_all_4D(split, 1, 2, group=group)
    finally:
        destroy_distributed_env()


@pytest.mark.parametrize("world_size", [2, 4])
def test_send_layouts_match_permuting_all_to_all(world_size: int):
    start_collective_workers(
        _run_send_layouts,
        args=(world_size, get_open_port()),
        nprocs=world_size,
        device_kind="cpu",
    )

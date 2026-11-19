# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Cross-clip tile schedule of the MiniMax-H3 temporal VAE forks.

``_TiledModel`` carries verbatim copies of the checkpoint's tiling methods
(``split_tiles``, ``blend``, ``tiled_encode``/``tiled_decode``, the
round-robin task split and the variable-shape gather, minus training-only
branches) around a cheap deterministic encoder/decoder. Four gloo ranks run
the forks with the per-clip schedule (``VLLM_OMNI_H3_VAE_GLOBAL_TILES=0``,
and the same unset, its default) and with the global one, round-robin
(``1``) and weighted: moments and uint8
frames must match bit for bit, the global schedule must not take the
per-clip gather at all, and each rank must run exactly its share.
"""

import math
import os
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from vllm_omni.diffusion.models.minimax_h3 import vae_global_tiles as global_tiles
from vllm_omni.diffusion.models.minimax_h3.vae_temporal import (
    _decode_temporal_streaming_uint8,
    _encode_temporal_tail_pad,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

ENV = global_tiles.MINIMAX_H3_VAE_GLOBAL_TILES_ENV
WORLD = 4
WEIGHTS = "1,1,0.5,1"  # rank 2 is "slow": it gets about half a share

# The checkpoint resolves its tile group through its module's
# ``get_parallel_state``; the global schedule must read the same dict.
_PARALLEL_STATE: dict = {}


def get_parallel_state():
    return _PARALLEL_STATE


def _all_gather_var_shape(tensor, group):
    """Forward half of the checkpoint's ``_AllGatherVarShape``."""
    tensor = tensor.contiguous()
    shape_info = torch.tensor(list(tensor.shape), dtype=torch.long, device=tensor.device)
    shape_list = [torch.empty_like(shape_info) for _ in range(dist.get_world_size(group=group))]
    dist.all_gather(shape_list, shape_info, group=group)
    all_shapes = [tuple(shape_tensor.tolist()) for shape_tensor in shape_list]
    flat_tensor = tensor.flatten()
    max_size = max(math.prod(s) for s in all_shapes)
    if flat_tensor.numel() < max_size:
        padded = torch.zeros(max_size, dtype=tensor.dtype, device=tensor.device)
        padded[: flat_tensor.numel()] = flat_tensor
        flat_tensor = padded
    gathered_flat = [torch.empty_like(flat_tensor) for _ in range(len(all_shapes))]
    dist.all_gather(gathered_flat, flat_tensor, group=group)
    return tuple(t[: math.prod(shape)].reshape(shape) for t, shape in zip(gathered_flat, all_shapes))


class _FakeDenorm:
    mean = (-2.11628412, -2.03571429, -1.80444444)
    std = (4.36681223, 4.46428571, 4.44444444)


class _FakeProcessor:
    transform_rev = _FakeDenorm()


class _TiledModel:
    """The checkpoint's tiling around a cheap deterministic codec."""

    def __init__(self, *, key_frame=False, stack_tiling=False):
        self.training = False
        self.vae_ratio = 2
        self.tile_size = self.decoder_tile_size = 8
        self.tile_overlap_min = self.decoder_tile_overlap_min = 2
        self.encoder_tiling = self.decoder_tiling = True
        self.encoder_parallel = self.decoder_parallel = False
        self.parallel_tiling = True
        self.stack_tiling = stack_tiling
        # encode_temporal contract
        self.clip_length = 3
        self.isolated_first_frame = False
        self.isolated_last_frame = False
        self.isolated_key_frame = key_frame
        self.frame_pre_padding = 0
        self.token_drop = 1
        self.frame_drop = 1
        # streaming-decode contract
        self.vae_ratio_t = 1
        self.tokens_chunk_size = 2
        self.token_overlap = 1
        self.frame_overlap = 1
        self.processor = _FakeProcessor()
        self.encode_calls = 0
        self.decode_calls = 0
        self.gathers = 0

    # -- deterministic codec: position-dependent so blending matters --------
    def encode(self, x):
        self.encode_calls += 1
        pooled = F.avg_pool3d(x, (1, self.vae_ratio, self.vae_ratio))
        return torch.cat([torch.sin(pooled * 1.7), torch.cos(pooled * 0.3 + 0.1)], dim=1)

    def decode(self, z):
        self.decode_calls += 1
        y = torch.sin(z[:, :3] * 2.3) * 0.6
        y = y.repeat_interleave(self.vae_ratio, dim=-1).repeat_interleave(self.vae_ratio, dim=-2)
        ramp = torch.linspace(-0.4, 0.4, y.shape[-1]).view(1, 1, 1, 1, -1)
        return y + ramp

    # -- verbatim checkpoint tiling (klvae.AutoencoderKL) -------------------
    def split_tiles(self, input_len, is_decoder=False):
        tile_size = self.decoder_tile_size if is_decoder else self.tile_size
        tile_overlap_min = self.decoder_tile_overlap_min if is_decoder else self.tile_overlap_min
        if tile_size >= input_len:
            return [0], [input_len], []
        N = math.ceil(input_len / tile_size)
        while True:
            overlaps = [tile_overlap_min] * (N - 1)
            remaining = tile_size * N - sum(overlaps) - input_len
            if remaining < 0:
                N += 1
            else:
                break
        remaining_units = remaining // self.vae_ratio
        for i in range(remaining_units):
            overlaps[i % (N - 1)] += self.vae_ratio
        tile_start_idx = [0]
        for i in range(N - 1):
            tile_start_idx.append(tile_start_idx[-1] + tile_size - overlaps[i])
        tile_len = [tile_size] * N
        return tile_start_idx, tile_len, overlaps

    def blend(self, a, b, blend_extent, dim):
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        weight_a = 1 - positions / blend_extent
        weight_b = positions / blend_extent
        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = weight_a.view(shape)
        weight_b = weight_b.view(shape)
        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        a_overlap = a[tuple(slice_a)]
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)
        b_overlap = b[tuple(slice_b)]
        blended = a_overlap * weight_a + b_overlap * weight_b
        if blend_extent < b.shape[dim]:
            slice_b_rest = [slice(None)] * b.ndim
            slice_b_rest[dim] = slice(blend_extent, None)
            return torch.cat([blended, b[tuple(slice_b_rest)]], dim=dim)
        return blended

    def _all_gather_tiled_results(self, tasks, num_tiles):
        self.gathers += 1
        state = get_parallel_state()
        group = state["sp_process_group"]
        sp_size = state["sp_size"]
        sp_rank = state["sp_rank"]
        if not tasks:
            raise ValueError(f"Found empty tasks on sp rank {sp_rank}")
        stacked = torch.stack(tasks, dim=0)
        gathered = _all_gather_var_shape(stacked, group=group)
        results = [None] * num_tiles
        for rank, rank_tensors in enumerate(gathered):
            for k in range(rank_tensors.shape[0]):
                global_idx = k * sp_size + rank
                if global_idx >= num_tiles:
                    break
                results[global_idx] = rank_tensors[k]
        return results

    def _local_tile_indices(self, num_tiles, sp_rank, sp_size):
        return list(range(sp_rank, num_tiles, sp_size))

    def _run_tile_tasks(self, tiles, tile_indices, forward_fn, stack_tiling):
        if stack_tiling and tile_indices:
            sample_batch_size = tiles[0].shape[0]
            tile_batch = torch.cat([tiles[idx] for idx in tile_indices], dim=0)
            output_batch = forward_fn(tile_batch)
            return list(output_batch.unflatten(0, (len(tile_indices), sample_batch_size)).unbind(dim=0))
        return [forward_fn(tiles[idx]) for idx in tile_indices]

    def _sp(self):
        if self.parallel_tiling:
            state = get_parallel_state()
            return state["sp_rank"], state["sp_size"]
        return 0, 1

    def _blend_rows(self, rows, y_overlap, x_overlap):
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend(rows[i - 1][j], tile, y_overlap[i - 1], dim=-2)
                if j > 0:
                    tile = self.blend(row[j - 1], tile, x_overlap[j - 1], dim=-1)
                if i < len(rows) - 1:
                    tile = tile[..., : -y_overlap[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, : -x_overlap[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def _tiled(self, x, tiles, i_max, j_max, forward_fn):
        sp_rank, sp_size = self._sp()
        local = self._local_tile_indices(len(tiles), sp_rank, sp_size)
        tasks = self._run_tile_tasks(tiles, local, forward_fn, self.stack_tiling)
        if sp_size > 1:
            dist.barrier(group=get_parallel_state()["sp_process_group"])
            tasks = self._all_gather_tiled_results(tasks, len(tiles))
        rows = [[None for _ in range(j_max)] for _ in range(i_max)]
        for idx, out in enumerate(tasks):
            rows[idx // j_max][idx % j_max] = out.to(x.device)
        return rows

    def tiled_encode(self, x):
        y_idx, y_len, y_overlap = self.split_tiles(x.shape[-2], False)
        x_idx, x_len, x_overlap = self.split_tiles(x.shape[-1], False)
        tiles = [x[..., i : i + n, j : j + m] for i, n in zip(y_idx, y_len) for j, m in zip(x_idx, x_len)]
        rows = self._tiled(x, tiles, len(y_idx), len(x_idx), self.encode)
        return self._blend_rows(
            rows, [o // self.vae_ratio for o in y_overlap], [o // self.vae_ratio for o in x_overlap]
        )

    def tiled_decode(self, z):
        r = self.vae_ratio
        y_idx, y_len, y_overlap = self.split_tiles(z.shape[-2] * r, True)
        x_idx, x_len, x_overlap = self.split_tiles(z.shape[-1] * r, True)
        tiles = [
            z[..., i // r : i // r + n // r, j // r : j // r + m // r]
            for i, n in zip(y_idx, y_len)
            for j, m in zip(x_idx, x_len)
        ]
        rows = self._tiled(z, tiles, len(y_idx), len(x_idx), self.decode)
        return self._blend_rows(rows, y_overlap, x_overlap)

    def _adaptive_encode(self, x):
        return self.tiled_encode(x) if self.encoder_tiling else self.encode(x)

    def _adaptive_decode(self, z):
        return self.tiled_decode(z) if self.decoder_tiling else self.decode(z)

    def _decode_temporal_pad_frames(self, z, pad_tokens):
        if pad_tokens <= 0:
            return 0
        intra_tail = self.clip_length % self.vae_ratio_t
        if intra_tail == 0:
            return int(pad_tokens) * int(self.vae_ratio_t)
        z_len_before_pad = z.shape[2] - pad_tokens
        return sum(
            (intra_tail if (z_len_before_pad + k) % self.tokens_chunk_size == 0 else self.vae_ratio_t)
            for k in range(pad_tokens)
        )

    def _decode_temporal_output_frame_plan(self, z, z_head, z_tail, num_chunks, pad_tokens):
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        split_count = int(self.token_drop > 0) + 1
        total_frames = 0
        final_overlap_frames = 0
        for i in range(num_chunks):
            t_start_idx = i * self.tokens_chunk_size
            t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
            clip_frame_len = max(0, min(t_end_idx, z.shape[2]) - min(t_start_idx, z.shape[2])) * self.vae_ratio_t
            for j in range(split_count):
                f_start_idx = j * chunk_dec
                f_end_idx = min(f_start_idx + chunk_dec, clip_frame_len)
                chunk_frames = max(0, f_end_idx - f_start_idx - self.frame_pre_padding)
                if j == 0:
                    total_frames += chunk_frames
                else:
                    final_overlap_frames = chunk_frames
        total_frames += final_overlap_frames
        pad_frames = self._decode_temporal_pad_frames(z, pad_tokens)
        return int(total_frames), int(pad_frames), int(total_frames - pad_frames)


def _video(frames, height=12, width=20):
    torch.manual_seed(1234)
    return torch.randn(1, 3, frames, height, width)


def _latent(tokens, height=6, width=10):
    torch.manual_seed(4321)
    return torch.randn(1, 3, tokens, height, width)


def _run(model, mode, fn):
    if mode is None:
        os.environ.pop(ENV, None)  # unset: the per-clip schedule
    else:
        os.environ[ENV] = mode
    model.encode_calls = model.decode_calls = model.gathers = 0
    out = fn()
    return out, (model.encode_calls, model.decode_calls, model.gathers)


def _global_tiles_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=WORLD, timeout=timedelta(seconds=60)
    )
    try:
        _PARALLEL_STATE.update(sp_rank=rank, sp_size=WORLD, sp_process_group=dist.group.WORLD)
        tiles_per_clip = 6  # 2 x 3 grid for 12x20 pixels and 6x10 latents
        per_clip_share = len(range(rank, tiles_per_clip, WORLD))

        # Encode: 11 frames -> 4 clips of 3 (the tail clip pads in-chunk).
        model = _TiledModel()
        x = _video(11)
        old, old_counts = _run(model, "0", lambda: _encode_temporal_tail_pad(model, x))
        unset, unset_counts = _run(model, None, lambda: _encode_temporal_tail_pad(model, x))
        assert torch.equal(unset, old) and unset_counts == old_counts
        new, new_counts = _run(model, "1", lambda: _encode_temporal_tail_pad(model, x))
        assert new.shape == old.shape and torch.equal(new, old)
        assert old_counts == (4 * per_clip_share, 0, 4)
        assert new_counts == (len(range(rank, 4 * tiles_per_clip, WORLD)), 0, 0)
        weighted, weighted_counts = _run(model, WEIGHTS, lambda: _encode_temporal_tail_pad(model, x))
        assert torch.equal(weighted, old)
        assert weighted_counts == ((7, 7, 3, 7)[rank], 0, 0)
        # Tile parallelism itself is exact: a rank-local encode agrees too.
        model.parallel_tiling = False
        local, _ = _run(model, "1", lambda: _encode_temporal_tail_pad(model, x))
        assert torch.equal(local, old)

        # Isolated key frames encode two shapes per clip: per-clip schedule.
        model = _TiledModel(key_frame=True)
        old, old_counts = _run(model, "0", lambda: _encode_temporal_tail_pad(model, x))
        new, new_counts = _run(model, "1", lambda: _encode_temporal_tail_pad(model, x))
        assert torch.equal(new, old) and new_counts == old_counts and new_counts[2] == 8

        # Streaming uint8 decode: 7 tokens -> 3 clips of 2 + 1 overlap token.
        for temporal_cat_dtype in (None, torch.float16):
            model = _TiledModel()
            z = _latent(7)

            def decode():
                return _decode_temporal_streaming_uint8(model, z, None, None, 3, 0, temporal_cat_dtype)

            old, old_counts = _run(model, "0", decode)
            unset, unset_counts = _run(model, None, decode)
            assert torch.equal(unset, old) and unset_counts == old_counts
            new, new_counts = _run(model, "1", decode)
            assert old.dtype == torch.uint8 and new.shape == old.shape and torch.equal(new, old)
            assert old_counts == (0, 3 * per_clip_share, 3)
            assert new_counts == (0, len(range(rank, 3 * tiles_per_clip, WORLD)), 0)
            weighted, weighted_counts = _run(model, WEIGHTS, decode)
            assert torch.equal(weighted, old)
            assert weighted_counts == (0, (5, 5, 3, 5)[rank], 0)
            model.parallel_tiling = False
            local, _ = _run(model, "1", decode)
            assert torch.equal(local, old)

        # Per-clip gathers return every clip's outputs in global order, with
        # per-clip shares that differ between ranks (3 clips of 5 tiles over
        # 4 ranks) and with weighted shares.
        tiles = [torch.full((1, 2), float(i)) for i in range(15)]
        for mode in ("1", WEIGHTS):
            os.environ[ENV] = mode
            run = global_tiles._GlobalTileRun(_PARALLEL_STATE, tiles, 5, lambda t: t * 2)
            gathered = [tile for clip in range(len(run)) for tile in run.gather(clip)]
            assert torch.equal(torch.stack(gathered), torch.stack(tiles) * 2)
            with pytest.raises(RuntimeError, match="already gathered"):
                run.gather(0)

        # A rank whose tile outputs differ makes every rank raise instead of
        # leaving the others in an all-gather of mismatched sizes.
        with pytest.raises(RuntimeError, match="differing tile outputs"):
            global_tiles._GlobalTileRun(_PARALLEL_STATE, tiles, 5, lambda t: torch.zeros(1, 3) if rank == 1 else t)
    finally:
        dist.destroy_process_group()


def test_global_schedule_matches_per_clip_schedule_bitwise(tmp_path):
    context = mp.spawn(_global_tiles_worker, args=(str(tmp_path / "rendezvous"),), nprocs=WORLD, join=False)
    deadline = time.monotonic() + 150
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("global tile workers did not finish within 150 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


# ---------------------------------------------------------------------------
# single-process: schedule arithmetic, switch, and declines
# ---------------------------------------------------------------------------


def _shares(owners, sp_size=8):
    return [owners.count(rank) for rank in range(sp_size)]


def test_canonical_request_balances_tiles():
    # 1344x768: 4 x 7 = 28 tiles per clip, 8 encode and 7 decode clips, 8 ranks.
    per_clip = [len(range(r, 28, 8)) for r in range(8)]
    assert per_clip == [4, 4, 4, 4, 3, 3, 3, 3]
    encode = global_tiles.global_tile_owners(224, 8)
    assert encode == [g % 8 for g in range(224)]  # the checkpoint's round-robin order
    assert _shares(encode) == [28] * 8
    assert _shares(global_tiles.global_tile_owners(196, 8)) == [25] * 4 + [24] * 4
    # One rank at ~0.83x speed (a clock-capped GPU) gets a proportional share.
    weights = (1, 1, 1, 1, 1, 0.83, 1, 1)
    assert _shares(global_tiles.global_tile_owners(224, 8, weights)) == [29, 29, 29, 29, 28, 24, 28, 28]
    assert _shares(global_tiles.global_tile_owners(196, 8, weights)) == [25, 25, 25, 25, 25, 21, 25, 25]


@pytest.mark.parametrize(
    "weights",
    [(1.0, 1.0, 1.0), (1.0, 1e-6, 1.0), (5.0, 1.0, 0.1), (0.3, 0.3, 0.3)],
)
@pytest.mark.parametrize("num_tiles", [3, 4, 7, 50])
def test_weighted_owners_cover_every_tile_and_rank(weights, num_tiles):
    owners = global_tiles.global_tile_owners(num_tiles, 3, weights)
    assert len(owners) == num_tiles
    assert all(count >= 1 for count in _shares(owners, 3))
    with pytest.raises(ValueError, match="cannot give each"):
        global_tiles.global_tile_owners(2, 3, weights)


def test_switch_values():
    parse = global_tiles._parse_global_tiles
    assert parse("0") == (False, None) and parse("off") == (False, None)
    assert parse("") == (False, None) and parse("  ") == (False, None)
    assert parse("1") == (True, None) and parse("on") == (True, None)
    assert parse(" 1,1,0.83 ") == (True, (1.0, 1.0, 0.83))
    for bad in ("fast", "1,,1", "1,0", "1,-2", "nan,1"):
        with pytest.raises(ValueError, match=ENV):
            parse(bad)


def test_switch_is_off_when_unset(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert global_tiles.vae_global_tiles_enabled() is False
    assert global_tiles._global_tile_weights(8) is None
    monkeypatch.setenv(ENV, "")
    assert global_tiles.vae_global_tiles_enabled() is False
    for value in ("1", "1,1,0.83"):
        monkeypatch.setenv(ENV, value)
        assert global_tiles.vae_global_tiles_enabled() is True


def test_weights_for_another_group_size_fall_back_to_equal_shares(monkeypatch):
    monkeypatch.setenv(ENV, "1,1,0.5")
    assert global_tiles._global_tile_weights(3) == (1.0, 1.0, 0.5)
    assert global_tiles._global_tile_weights(8) is None


@pytest.fixture
def parallel_state():
    _PARALLEL_STATE.clear()
    _PARALLEL_STATE.update(sp_rank=0, sp_size=WORLD, sp_process_group=object())
    yield _PARALLEL_STATE
    _PARALLEL_STATE.clear()


@pytest.mark.parametrize("decoder", [False, True])
def test_switch_and_configuration_gate_the_schedule(monkeypatch, parallel_state, decoder):
    monkeypatch.delenv(ENV, raising=False)
    assert global_tiles.global_tile_state(_TiledModel(), decoder=decoder) is None
    monkeypatch.setenv(ENV, "0")
    assert global_tiles.global_tile_state(_TiledModel(), decoder=decoder) is None
    monkeypatch.setenv(ENV, "1")
    assert global_tiles.global_tile_state(_TiledModel(), decoder=decoder) is parallel_state

    def declined(**attrs):
        model = _TiledModel()
        for name, value in attrs.items():
            setattr(model, name, value)
        return global_tiles.global_tile_state(model, decoder=decoder) is None

    assert declined(stack_tiling=True)
    assert declined(training=True)
    assert declined(parallel_tiling=False)
    assert declined(**{"decoder_tiling" if decoder else "encoder_tiling": False})
    assert declined(**{"decoder_parallel" if decoder else "encoder_parallel": True})
    assert declined(_all_gather_tiled_results=lambda tasks, num_tiles: tasks)
    assert declined(**{"tiled_decode" if decoder else "tiled_encode": lambda x: x})
    parallel_state["sp_size"] = 1
    assert declined()


def test_encoder_cls_token_aggregation_keeps_per_clip_schedule(monkeypatch, parallel_state):
    monkeypatch.setenv(ENV, "1")
    model = _TiledModel()
    model.encoder = type("Encoder", (), {"loss_info": {}})()
    assert global_tiles.global_tile_state(model, decoder=False) is None
    assert global_tiles.global_tile_state(model, decoder=True) is parallel_state


def test_uneven_tile_grid_declines_before_any_tile_runs(parallel_state):
    # 11 rows split into tiles of 8 starting at 0 and 4: the last tile is
    # clipped to 7 rows, so shares would mix shapes.
    model = _TiledModel()
    clips = [torch.zeros(1, 3, 3, 11, 20) for _ in range(2)]
    assert global_tiles.global_tiled_encode(model, parallel_state, clips) is None
    assert model.encode_calls == 0
    mismatched = [torch.zeros(1, 3, 3, 12, 20), torch.zeros(1, 3, 2, 12, 20)]
    assert global_tiles.global_tiled_encode(model, parallel_state, mismatched) is None
    latents = [torch.zeros(1, 3, 3, 6, 10), torch.zeros(1, 3, 2, 6, 10)]
    assert global_tiles.global_tiled_decode(model, parallel_state, latents) is None
    assert model.decode_calls == 0


def test_decoded_clips_assemble_in_order_from_per_clip_gathers():
    class _Run:
        def __init__(self):
            self.gathered = []

        def gather(self, clip):
            self.gathered.append(clip)
            return [torch.full((1, 3, 1, 8, 8), float(2 * clip + k)) for k in range(2)]

    run = _Run()
    clips = global_tiles.GlobalDecodedClips(_TiledModel(), run, 2, [], [2], [torch.device("cpu")] * 2)
    assert len(clips) == 2
    with pytest.raises(RuntimeError, match="in order"):
        clips.assemble(1)
    assert clips.assemble(0).shape == (1, 3, 1, 8, 14)
    with pytest.raises(RuntimeError, match="in order"):
        clips.assemble(0)
    clips.assemble(1)
    assert run.gathered == [0, 1]

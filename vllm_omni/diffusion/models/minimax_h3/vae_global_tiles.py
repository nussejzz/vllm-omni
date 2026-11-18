# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One tile schedule across all temporal clips of the MiniMax-H3 video VAE.

The checkpoint's ``tiled_encode`` / ``tiled_decode`` parallelize one temporal
clip at a time: the clip's tiles go round-robin over the tile group
(``range(sp_rank, num_tiles, sp_size)``), then every rank meets at a
``dist.barrier`` and a variable-shape all-gather before blending. A
1344x768 clip has 28 tiles, so ranks 0-3 run 4 tiles and ranks 4-7 run 3 and
then wait at the barrier, once for each of the 8 encode and 7 decode clips
of a 124-frame request.

The clips are independent. ``encode_temporal`` hands each clip to
``_adaptive_encode`` with no carried state (the causal CNN encoder pads every
call and keeps no feature cache), and the streaming decode blends the
temporal overlap only after a clip is decoded, so nothing flows from one
clip's tiles into another clip's. This module therefore lays the tiles of
all clips out as one list, clip-major and row-major inside a clip, and hands
it to the group round-robin: 224 encode tiles become 28 per rank and 196
decode tiles at most 25 per rank. Every rank runs all its tiles back to
back; only then are the outputs gathered, one equal-shape all-gather per
clip (``_GlobalTileRun``), and each clip is blended from its tiles with the
checkpoint's own code in the original order. Each tile still runs alone
through the checkpoint's ``encode`` / ``decode`` on the same input view, and
the gather only copies, so latents and decoded frames are bit-identical to
the per-clip schedule. The extra residency is this rank's own outputs until
they are gathered: 25 + 4 decoded fp32 tiles of 3x28x256x256 (0.6 GiB) on
the canonical request, about 7 MB for the reference encode.

Every eligibility decision below is a pure function of the input shapes, the
checkpoint configuration and the environment, so all ranks take the same
path and enter the same collectives. Anything the per-clip schedule handles
that this one does not mirror exactly (stacked tiling, isolated key frames,
clips or tiles of differing shapes, encoder/decoder sequence parallelism,
cls-token aggregation, a gather routed through an adapter workspace) keeps
the per-clip schedule.

``VLLM_OMNI_H3_VAE_GLOBAL_TILES`` selects the schedule: ``0`` keeps the
per-clip schedule, unset or ``1`` hands the global list out round-robin, and
a comma-separated list of positive per-rank weights (one per tile-group
rank, e.g. ``1,1,1,1,1,0.83,1,1``) hands rank ``r`` a share of the tiles
proportional to its weight. Weights exist for groups whose GPUs are not
equally fast: a rank whose clock is capped would otherwise set the pace of
the whole round-robin. They move tiles between ranks only; a tile runs the
same kernels on whichever rank owns it, so on identical GPUs the output does
not depend on them.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import init_logger

logger = init_logger(__name__)

MINIMAX_H3_VAE_GLOBAL_TILES_ENV = "VLLM_OMNI_H3_VAE_GLOBAL_TILES"


def _parse_global_tiles(value: str) -> tuple[bool, tuple[float, ...] | None]:
    """``(enabled, per-rank weights or None)`` for one value of the switch."""
    value = value.strip().lower()
    if value in ("0", "false", "off", "no"):
        return False, None
    if value in ("", "1", "true", "on", "yes"):
        return True, None
    try:
        weights = tuple(float(item) for item in value.split(","))
    except ValueError:
        weights = ()
    if not weights or not all(math.isfinite(w) and w > 0 for w in weights):
        raise ValueError(
            f"{MINIMAX_H3_VAE_GLOBAL_TILES_ENV} must be 0, 1, or comma-separated positive per-rank weights, "
            f"got {value!r}"
        )
    return True, weights


def vae_global_tiles_enabled() -> bool:
    """Whether the cross-clip tile schedule is on (default) or rolled back (``0``)."""
    return _parse_global_tiles(os.environ.get(MINIMAX_H3_VAE_GLOBAL_TILES_ENV, "1"))[0]


# Reject a malformed switch when the model is imported rather than at the
# first request; requests still read it, so it can change between them.
_parse_global_tiles(os.environ.get(MINIMAX_H3_VAE_GLOBAL_TILES_ENV, "1"))


def _global_tile_weights(sp_size: int) -> tuple[float, ...] | None:
    weights = _parse_global_tiles(os.environ.get(MINIMAX_H3_VAE_GLOBAL_TILES_ENV, "1"))[1]
    if weights is not None and len(weights) != sp_size:
        logger.warning_once(
            "%s lists %d rank weights but the VAE tile group has %d ranks; using equal shares",
            MINIMAX_H3_VAE_GLOBAL_TILES_ENV,
            len(weights),
            sp_size,
        )
        return None
    return weights


def global_tile_owners(num_tiles: int, sp_size: int, weights: Sequence[float] | None = None) -> list[int]:
    """The rank that runs each global tile.

    Without weights this is the checkpoint's round-robin (tile ``g`` on rank
    ``g % sp_size``). With weights each rank's share is its largest-remainder
    quota of ``num_tiles`` (at least one tile), handed out by a round-robin
    that skips ranks whose quota is used up. A pure function of its
    arguments, so every rank derives the same owners.
    """
    if num_tiles < sp_size:
        raise ValueError(f"{num_tiles} tiles cannot give each of {sp_size} ranks a tile")
    if weights is None:
        return [index % sp_size for index in range(num_tiles)]
    total = float(sum(weights))
    exact = [num_tiles * float(w) / total for w in weights]
    quotas = [max(1, int(share)) for share in exact]
    by_remainder = sorted(range(sp_size), key=lambda r: (-(exact[r] - int(exact[r])), r))
    while sum(quotas) < num_tiles:
        for rank in by_remainder:
            if sum(quotas) == num_tiles:
                break
            quotas[rank] += 1
    while sum(quotas) > num_tiles:  # only when the one-tile floor lifted a tiny share
        rank = max(range(sp_size), key=lambda r: (quotas[r], -r))
        quotas[rank] -= 1
    owners: list[int] = []
    rank = 0
    while len(owners) < num_tiles:
        if quotas[rank] > 0:
            owners.append(rank)
            quotas[rank] -= 1
        rank = (rank + 1) % sp_size
    return owners


def global_tile_state(model: Any, *, decoder: bool) -> dict[str, Any] | None:
    """The checkpoint's tile-group state when the global schedule may run, else None.

    Only configuration is inspected here (shapes are checked per call), and
    the parallel state is the very dict the checkpoint's ``tiled_*`` methods
    read, so the verdict is the same on every rank.
    """
    if not vae_global_tiles_enabled():
        return None
    if bool(getattr(model, "training", False)) or bool(getattr(model, "stack_tiling", False)):
        # Stacked tiling batches a rank's tiles into one forward; batching
        # across clips would change the GEMM shapes and therefore the numbers.
        return None
    if not bool(getattr(model, "decoder_tiling" if decoder else "encoder_tiling", False)):
        return None
    if bool(getattr(model, "decoder_parallel" if decoder else "encoder_parallel", False)):
        # Sequence-parallel encode/decode runs collectives inside every tile.
        return None
    if not decoder and hasattr(getattr(model, "encoder", None), "loss_info"):
        # The checkpoint aggregates encoder cls tokens per tiled call.
        return None
    overrides = ("_adaptive_decode", "tiled_decode") if decoder else ("_adaptive_encode", "tiled_encode")
    if any(name in vars(model) for name in overrides):
        # This module mirrors the checkpoint's own tiled methods; an instance
        # override would be bypassed.
        return None
    if "_all_gather_tiled_results" in vars(model):
        # The adapter routes gathers through a persistent workspace (XPU,
        # whose collectives must land at a stable address); this schedule
        # allocates its own landing buffer, so keep the per-clip gathers.
        return None
    if not bool(getattr(model, "parallel_tiling", False)):
        return None
    getter = getattr(sys.modules.get(type(model).__module__), "get_parallel_state", None)
    if not callable(getter):
        return None
    state = getter()
    if int(state.get("sp_size", 1) or 1) <= 1 or state.get("sp_process_group") is None:
        return None
    return state


def _uniform_extent(starts: Sequence[int], lengths: Sequence[int], axis_len: int) -> bool:
    # Python slicing clips the last tile at the axis end; equal extents keep
    # every tile, and so every gathered output, one shape.
    return len({min(int(s) + int(n), axis_len) - int(s) for s, n in zip(starts, lengths)}) == 1


def _blend_tile_grid(
    model: Any,
    tiles: Sequence[torch.Tensor],
    j_max: int,
    y_overlap: Sequence[int],
    x_overlap: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """The checkpoint's ``tiled_encode`` / ``tiled_decode`` blend, verbatim."""
    i_max = len(tiles) // j_max
    rows: list[list[Any]] = [[None for _ in range(j_max)] for _ in range(i_max)]
    for idx, tile in enumerate(tiles):
        i, j = idx // j_max, idx % j_max
        rows[i][j] = tile.to(device)

    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = model.blend(rows[i - 1][j], tile, y_overlap[i - 1], dim=-2)
            if j > 0:
                tile = model.blend(row[j - 1], tile, x_overlap[j - 1], dim=-1)
            if i < len(rows) - 1:
                tile = tile[..., : -y_overlap[i], :]
            if j < len(row) - 1:
                tile = tile[..., :, : -x_overlap[j]]
            result_row.append(tile)
        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=-2)


_MAX_TILE_NDIM = 8


class _GlobalTileRun:
    """This rank's share of a clip-major tile list, run up front and gathered clip by clip.

    Ownership comes from ``global_tile_owners`` (the checkpoint's round-robin
    unless per-rank weights are set). The constructor runs every tile this
    rank owns, one per forward, back to back with no collective in between,
    and copies each output into a local send buffer as soon as it exists.
    ``gather(c)`` then returns clip ``c``'s outputs in the order the
    checkpoint's ``_all_gather_tiled_results`` returns them for that clip.

    The callers only get here for an equal-shape tile grid, so every output
    has one shape and a clip's gather is one ``all_gather_into_tensor``
    straight from a contiguous slice of the send buffer: a rank's tiles of
    clip ``c`` are contiguous there because it runs its tiles in global
    order, and the slice is padded to the largest per-rank share of the clip
    (the padding is never read back). Gathering per clip after all the
    compute keeps the balance of one global schedule (the first gather is
    the only point where ranks wait for each other) while only one clip's
    outputs from every rank are resident at a time, so the decode working
    set stays near the per-clip schedule's instead of holding every decoded
    tile of the request at once.

    Before the first gather the ranks exchange their output shapes in a
    fixed-size message and all raise together on a mismatch, so a rank whose
    tiles came out differently cannot leave its peers waiting in an
    all-gather of mismatched sizes.
    """

    def __init__(
        self,
        state: dict[str, Any],
        tiles: Sequence[torch.Tensor],
        tiles_per_clip: int,
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self._group = state["sp_process_group"]
        self._sp_rank = sp_rank = int(state["sp_rank"])
        self._sp_size = sp_size = int(state["sp_size"])
        num_tiles = len(tiles)
        if tiles_per_clip <= 0 or num_tiles % tiles_per_clip:
            raise ValueError(f"{num_tiles} tiles do not split into clips of {tiles_per_clip}")
        self._per_clip = tiles_per_clip
        self._num_clips = num_tiles // tiles_per_clip
        self._owners = owners = global_tile_owners(num_tiles, sp_size, _global_tile_weights(sp_size))
        # Position of every tile in its owner's send buffer, and where each
        # rank's share of each clip starts there.
        self._position: list[int] = []
        taken = [0] * sp_size
        self._clip_start: list[list[int]] = []
        self._clip_slots: list[int] = []
        for clip in range(self._num_clips):
            self._clip_start.append(list(taken))
            for owner in owners[clip * tiles_per_clip : (clip + 1) * tiles_per_clip]:
                self._position.append(taken[owner])
                taken[owner] += 1
            self._clip_slots.append(max(t - s for t, s in zip(taken, self._clip_start[clip])))

        send: torch.Tensor | None = None
        consistent = True
        for index in (index for index, owner in enumerate(owners) if owner == sp_rank):
            out = forward_fn(tiles[index])
            if send is None:
                send = out.new_empty((taken[sp_rank] + max(self._clip_slots), *out.shape))
            if out.shape != send.shape[1:] or out.dtype != send.dtype:
                consistent = False
            else:
                send[self._position[index]].copy_(out)
            del out
        if send is None:
            raise ValueError(f"MiniMax-H3 global VAE tiles left sp rank {sp_rank} without a tile")

        tile_shape = list(send.shape[1:])
        if len(tile_shape) > _MAX_TILE_NDIM:
            raise ValueError(f"MiniMax-H3 global VAE tiles support at most {_MAX_TILE_NDIM}-d tiles, got {tile_shape}")
        message = torch.tensor(
            [int(consistent), send.element_size(), len(tile_shape), *tile_shape]
            + [0] * (_MAX_TILE_NDIM - len(tile_shape)),
            dtype=torch.int64,
            device=send.device,
        )
        messages = message.new_empty((sp_size, message.numel()))
        # Flat views: every backend accepts a 1-d output that concatenates the inputs.
        dist.all_gather_into_tensor(messages.view(-1), message, group=self._group)
        if not bool((messages == messages[0]).all()) or not bool(messages[:, 0].all()):
            raise RuntimeError(
                "MiniMax-H3 global VAE tiles produced differing tile outputs across ranks: "
                f"(consistent, itemsize, ndim, shape...) per rank = {messages.tolist()}"
            )
        self._send: torch.Tensor | None = send

    def __len__(self) -> int:
        return self._num_clips

    def gather(self, clip: int) -> list[torch.Tensor]:
        """Clip ``clip``'s tile outputs from every rank, in global tile order.

        Every rank must gather the same clips in the same order. The send
        buffer is released with the last clip.
        """
        send = self._send
        if send is None:
            raise RuntimeError("MiniMax-H3 global VAE tiles were already gathered")
        slots = self._clip_slots[clip]
        start = self._clip_start[clip][self._sp_rank]
        landing = send.new_empty((self._sp_size, slots, *send.shape[1:]))
        dist.all_gather_into_tensor(landing.view(-1), send[start : start + slots].reshape(-1), group=self._group)
        begin = clip * self._per_clip
        tiles = [
            landing[owner, self._position[index] - self._clip_start[clip][owner]]
            for index, owner in enumerate(self._owners[begin : begin + self._per_clip], start=begin)
        ]
        if clip == self._num_clips - 1:
            self._send = None
        return tiles


def global_tiled_encode(model: Any, state: dict[str, Any], clips: Sequence[torch.Tensor]) -> list[torch.Tensor] | None:
    """``[model.tiled_encode(clip) for clip in clips]`` under one tile schedule.

    Returns None, before any tile runs, when the clips do not share one shape
    or their tile grid is not equal-shaped; the caller then keeps the
    per-clip schedule.
    """
    if len(clips) < 2 or any(clip.shape != clips[0].shape for clip in clips[1:]):
        return None
    height, width = int(clips[0].shape[-2]), int(clips[0].shape[-1])
    y_idx, y_len, y_overlap = model.split_tiles(height, False)
    x_idx, x_len, x_overlap = model.split_tiles(width, False)
    if not (_uniform_extent(y_idx, y_len, height) and _uniform_extent(x_idx, x_len, width)):
        return None
    num_tiles = len(y_idx) * len(x_idx)
    if len(clips) * num_tiles < int(state["sp_size"]):
        return None
    tiles = [
        clip[..., i_pos : i_pos + i_len, j_pos : j_pos + j_len]
        for clip in clips
        for i_pos, i_len in zip(y_idx, y_len)
        for j_pos, j_len in zip(x_idx, x_len)
    ]
    run = _GlobalTileRun(state, tiles, num_tiles, model.encode)
    del tiles
    latent_y_overlap = [tile_overlap // model.vae_ratio for tile_overlap in y_overlap]
    latent_x_overlap = [tile_overlap // model.vae_ratio for tile_overlap in x_overlap]
    return [
        _blend_tile_grid(model, run.gather(c), len(x_idx), latent_y_overlap, latent_x_overlap, clip.device)
        for c, clip in enumerate(clips)
    ]


class GlobalDecodedClips:
    """Decoded tiles of every clip; ``assemble(i)`` gathers and blends clip ``i``.

    ``assemble(i)`` returns what ``model.tiled_decode(clip_zs[i])`` returns.
    Every rank must assemble the clips in the same order (the streaming
    decode loop does, clip 0 first), and each clip exactly once.
    """

    def __init__(
        self,
        model: Any,
        run: _GlobalTileRun,
        j_max: int,
        y_overlap: Sequence[int],
        x_overlap: Sequence[int],
        devices: Sequence[torch.device],
    ) -> None:
        self._model = model
        self._run = run
        self._j_max = j_max
        self._y_overlap = list(y_overlap)
        self._x_overlap = list(x_overlap)
        self._devices = list(devices)
        self._next = 0

    def __len__(self) -> int:
        return len(self._devices)

    def assemble(self, clip_index: int) -> torch.Tensor:
        if clip_index != self._next:
            raise RuntimeError(
                f"MiniMax-H3 global decode clips must be assembled in order: expected clip {self._next}, "
                f"got {clip_index}"
            )
        self._next += 1
        return _blend_tile_grid(
            self._model,
            self._run.gather(clip_index),
            self._j_max,
            self._y_overlap,
            self._x_overlap,
            self._devices[clip_index],
        )


def global_tiled_decode(
    model: Any, state: dict[str, Any], clip_zs: Sequence[torch.Tensor]
) -> GlobalDecodedClips | None:
    """Decode the tiles of every clip latent under one tile schedule.

    Returns None, before any tile runs, when the clip latents do not share
    one shape or their tile grid is not equal-shaped; the caller then keeps
    the per-clip schedule.
    """
    if len(clip_zs) < 2 or any(clip_z.shape != clip_zs[0].shape for clip_z in clip_zs[1:]):
        return None
    ratio = int(model.vae_ratio)
    latent_h, latent_w = int(clip_zs[0].shape[-2]), int(clip_zs[0].shape[-1])
    y_idx, y_len, y_overlap = model.split_tiles(latent_h * ratio, True)
    x_idx, x_len, x_overlap = model.split_tiles(latent_w * ratio, True)
    # The checkpoint slices decoder tiles in latent units.
    y_tiles = [(i_pos // ratio, i_len // ratio) for i_pos, i_len in zip(y_idx, y_len)]
    x_tiles = [(j_pos // ratio, j_len // ratio) for j_pos, j_len in zip(x_idx, x_len)]
    if not (
        _uniform_extent([p for p, _ in y_tiles], [n for _, n in y_tiles], latent_h)
        and _uniform_extent([p for p, _ in x_tiles], [n for _, n in x_tiles], latent_w)
    ):
        return None
    num_tiles = len(y_tiles) * len(x_tiles)
    if len(clip_zs) * num_tiles < int(state["sp_size"]):
        return None
    tiles = [
        clip_z[..., i_pos : i_pos + i_len, j_pos : j_pos + j_len]
        for clip_z in clip_zs
        for i_pos, i_len in y_tiles
        for j_pos, j_len in x_tiles
    ]
    run = _GlobalTileRun(state, tiles, num_tiles, model.decode)
    del tiles
    return GlobalDecodedClips(model, run, len(x_tiles), y_overlap, x_overlap, [clip_z.device for clip_z in clip_zs])


__all__ = [
    "MINIMAX_H3_VAE_GLOBAL_TILES_ENV",
    "GlobalDecodedClips",
    "global_tile_owners",
    "global_tile_state",
    "global_tiled_decode",
    "global_tiled_encode",
    "vae_global_tiles_enabled",
]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded H3 windows with synchronized latent-tail guides and global AV RoPE positions.

The guide/discard/append algorithm follows ComfyUI-Minimax-H3-Continuation:
https://github.com/ttulttul/ComfyUI-Minimax-H3-Continuation
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.models.minimax_h3.continuation_request import resolve_continuation_request
from vllm_omni.model_executor.models.minimax_h3.long_video import resolve_long_video_mode

from .packed_tokens import minimax_h3_pack_audio_latent, minimax_h3_patchify_video_latent

logger = init_logger(__name__)


def _video_t(frames: int) -> int:
    return (frames - 5) // 17 * 5 + 2


def _audio_t(frame_boundary: int) -> int:
    return round(frame_boundary * 5 / 3)


@dataclass(frozen=True)
class ContinuationWindow:
    start: int
    end: int
    overlap: int

    @property
    def audio_start(self) -> int:
        return _audio_t(self.start)

    @property
    def audio_end(self) -> int:
        return _audio_t(self.end)

    @property
    def overlap_audio_t(self) -> int:
        return _audio_t(self.start + self.overlap) - self.audio_start


@dataclass(frozen=True)
class LatentPrefix:
    """Clean latents of an earlier clip that the request's first frames repeat.

    ``start_frame`` places the request on that clip's global frame timeline;
    ``video``/``audio`` are its last ``overlap_frames`` frames of latents. They
    ride as a frame-0 AV timeline guide for every task: it anchors the window to
    the clip far more tightly than a Ref2VA ``latent_guide`` reference.
    """

    video: torch.Tensor  # [1, C, _video_t(overlap_frames), H, W]
    audio: torch.Tensor  # [2, C, overlap audio latents]
    start_frame: int
    overlap_frames: int


SEAM_PINS = ("none", "all", "chunk", "only")
SEAM_CLOCKS = ("global", "window")
SEAM_LOOKAHEADS = (0, 17, 34)
SEAM_MIN_SEED_STRIDE = 10007
_CHUNK_FRAMES = 17
_CHUNK_LATENTS = 5


@dataclass(frozen=True)
class SeamOptions:
    pin: str = "none"
    seed_stride: int = 0
    clock: str = "global"
    lookahead: int = 0
    return_overlap: bool = False
    diag: bool = False


def resolve_seam(
    extra: Mapping[str, Any],
    *,
    continuation: tuple[int, int] | None,
    renoise: bool,
    step_execution: bool,
) -> SeamOptions | None:
    raw = extra.get("h3_seam")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise OmniClientError("MiniMax H3 h3_seam must be an object")
    unknown = sorted(str(key) for key in set(raw) - set(SeamOptions.__dataclass_fields__))
    if unknown:
        raise OmniClientError(f"MiniMax H3 h3_seam has unknown keys: {', '.join(unknown)}")
    request = resolve_continuation_request(extra)
    chain = request is not None
    if continuation is None and not chain:
        raise OmniClientError("MiniMax H3 h3_seam requires long_video_mode=continuation or continue_from")
    if step_execution:
        raise OmniClientError("MiniMax H3 h3_seam requires request execution (not step execution)")
    seam = SeamOptions(**raw)
    if seam.pin not in SEAM_PINS:
        raise OmniClientError(f"MiniMax H3 h3_seam.pin must be one of {', '.join(SEAM_PINS)}")
    if seam.pin != "none" and not renoise:
        raise OmniClientError("MiniMax H3 h3_seam.pin requires the re-noising sampler (Pika H3 Flex)")
    if seam.pin in ("chunk", "only") and (continuation is None or chain):
        raise OmniClientError(
            f"MiniMax H3 h3_seam.pin={seam.pin} requires in-request continuation without continue_from"
        )
    if seam.pin == "chunk" and continuation[1] < 22:
        raise OmniClientError("MiniMax H3 h3_seam.pin=chunk requires continuation_overlap_frames >= 22")
    stride = seam.seed_stride
    valid = isinstance(stride, int) and not isinstance(stride, bool)
    if not valid or (stride and not SEAM_MIN_SEED_STRIDE <= stride < 2**31):
        raise OmniClientError(
            f"MiniMax H3 h3_seam.seed_stride must be 0 or an integer in [{SEAM_MIN_SEED_STRIDE}, 2**31)"
        )
    if stride and continuation is None:
        raise OmniClientError(
            "MiniMax H3 h3_seam.seed_stride applies to in-request windows; send each continue_from request its own seed"
        )
    if seam.clock not in SEAM_CLOCKS:
        raise OmniClientError("MiniMax H3 h3_seam.clock must be global or window")
    if seam.clock == "window" and continuation is not None and not stride:
        raise OmniClientError("MiniMax H3 h3_seam.clock=window requires h3_seam.seed_stride for in-request windows")
    lookahead = seam.lookahead
    if isinstance(lookahead, bool) or not isinstance(lookahead, int) or lookahead not in SEAM_LOOKAHEADS:
        raise OmniClientError("MiniMax H3 h3_seam.lookahead must be 0, 17 or 34")
    if continuation is not None and continuation[0] - continuation[1] - lookahead < 17:
        raise OmniClientError(
            "MiniMax H3 continuation window must exceed its overlap plus h3_seam.lookahead by 17 frames"
        )
    if continuation is not None and chain and continuation[0] - request.overlap_frames - lookahead < 17:
        raise OmniClientError(
            "MiniMax H3 continuation window must exceed continue_overlap_frames plus h3_seam.lookahead by 17 frames"
        )
    for name in ("return_overlap", "diag"):
        if not isinstance(getattr(seam, name), bool):
            raise OmniClientError(f"MiniMax H3 h3_seam.{name} must be a boolean")
    if seam.return_overlap and not chain:
        raise OmniClientError("MiniMax H3 h3_seam.return_overlap requires continue_from")
    return seam


def resolve_continuation(extra: Mapping[str, Any], *, task: str, step_execution: bool) -> tuple[int, int] | None:
    mode = resolve_long_video_mode(extra, task)
    if mode == "full":
        return None
    if task != "ref2va" or step_execution:
        raise OmniClientError("MiniMax H3 continuation requires Ref2VA request execution (not step execution)")
    window = extra.get("continuation_window_frames", 277)
    overlap = extra.get("continuation_overlap_frames", 22)
    for name, value in (("window", window), ("overlap", overlap)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 5 or value % 17 != 5:
            raise OmniClientError(f"MiniMax H3 continuation {name} must be an integer on the 17n+5 frame grid")
    if not 107 <= window <= 345 or overlap >= window:
        raise OmniClientError("MiniMax H3 continuation window must be 107..345 frames and exceed its overlap")
    return window, overlap


def plan_continuation_windows(
    total_frames: int,
    window_frames: int,
    overlap_frames: int,
    *,
    first_overlap: int = 0,
    origin: int = 0,
    lookahead: int = 0,
) -> list[ContinuationWindow]:
    """Windows over ``total_frames``, offset onto a global timeline by ``origin``.

    ``first_overlap`` hides the first frames of the first window behind a
    :class:`LatentPrefix`; later windows overlap their predecessor.
    """
    if total_frames < 5 or total_frames % 17 != 5:
        raise ValueError("total_frames must satisfy 17n+5")
    if not 5 <= overlap_frames < window_frames or window_frames % 17 != 5 or overlap_frames % 17 != 5:
        raise ValueError("window and overlap must satisfy 17n+5, with 5 <= overlap < window")
    if lookahead % 17 or not 0 <= lookahead <= window_frames - overlap_frames - 17:
        raise ValueError("lookahead must be a multiple of 17 that leaves new frames in every window")
    end = min(total_frames, window_frames)
    if first_overlap and (first_overlap % 17 != 5 or first_overlap >= end):
        raise ValueError("first_overlap must satisfy 17n+5 and leave new frames in the first window")
    windows = [ContinuationWindow(origin, origin + end, first_overlap)]
    while end < total_frames:
        start = end - overlap_frames - lookahead
        end = min(total_frames, start + window_frames)
        windows.append(ContinuationWindow(origin + start, origin + end, overlap_frames))
    return windows


def latent_guide_args(
    kwargs: Mapping[str, Any],
    video_tail: torch.Tensor,
    audio_tail: torch.Tensor,
    *,
    timeline_guide: bool = False,
) -> dict[str, Any]:
    """Add clean video/audio tail latents covering the window's first overlap frames.

    ``video_tail`` is ``[1, C, overlap_v, H, W]`` on the target canvas and
    ``audio_tail`` is ``[2, C, overlap_audio_t]``. Ref2VA appends them as the
    final ``latent_guide`` reference block; other tasks prepend a frame-0 AV
    timeline guide, which shares that origin and packs ahead of references.
    """
    overlap_v, overlap_audio_t = int(video_tail.shape[2]), int(audio_tail.shape[-1])
    if tuple(video_tail.shape[-2:]) != (kwargs["latent_h"], kwargs["latent_w"]):
        raise OmniClientError("MiniMax H3 latent guide must match the target canvas")
    guide_video = minimax_h3_patchify_video_latent(video_tail, patch_size=(1, 2, 2))
    guide_audio = minimax_h3_pack_audio_latent(audio_tail)
    shape = (overlap_v, kwargs["latent_h"], kwargs["latent_w"])
    if timeline_guide:
        block = {"kind": "video_audio", "frame_index": 0, "ref_audio_t": overlap_audio_t}
        return {
            "guide_blocks": [
                {**block, "latent_t": overlap_v, "latent_h": shape[1], "latent_w": shape[2]},
                *(kwargs.get("guide_blocks") or []),
            ],
            "visual_condition": _join(guide_video, kwargs.get("visual_condition"), first=True),
            "audio_condition": _join(guide_audio, kwargs.get("audio_condition"), first=True),
            "visual_condition_shapes": [shape, *(kwargs.get("visual_condition_shapes") or [])],
            "audio_condition_lengths": [overlap_audio_t, *(kwargs.get("audio_condition_lengths") or [])],
        }
    block = {"kind": "latent_guide", "ref_audio_t": overlap_audio_t}
    return {
        "ref_blocks": [
            *(kwargs.get("ref_blocks") or []),
            {**block, "latent_t": overlap_v, "latent_h": shape[1], "latent_w": shape[2]},
        ],
        "visual_condition": _join(guide_video, kwargs.get("visual_condition"), first=False),
        "audio_condition": _join(guide_audio, kwargs.get("audio_condition"), first=False),
        "visual_condition_shapes": [*(kwargs.get("visual_condition_shapes") or []), shape],
        "audio_condition_lengths": [*(kwargs.get("audio_condition_lengths") or []), overlap_audio_t],
    }


def _join(guide: torch.Tensor, existing: torch.Tensor | None, *, first: bool) -> torch.Tensor:
    if existing is None:
        return guide
    return torch.cat((guide, existing) if first else (existing, guide), dim=0)


def window_args(
    kwargs: Mapping[str, Any],
    window: ContinuationWindow,
    *,
    video_tail: torch.Tensor | None = None,
    audio_tail: torch.Tensor | None = None,
    timeline_guide: bool = False,
    audio_origin: int = 0,
) -> dict[str, Any]:
    """Denoise arguments for one window: global AV clock, window lengths and any tail guide."""
    frames = window.end - window.start
    args = {
        **kwargs,
        "num_frames": frames,
        "latent_t": _video_t(frames),
        "audio_t": window.audio_end - window.audio_start,
        # Keep fractional RoPE units; audio slice indices alone are rounded.
        "temporal_offset": window.start * (40.0 / 24.0),
    }
    source = kwargs.get("locked_audio_rows")
    if source is not None:
        source = source.reshape(2, kwargs["audio_t"], 32)
        args["locked_audio_rows"] = source[
            :, window.audio_start - audio_origin : window.audio_end - audio_origin
        ].reshape(-1, 32)
    if video_tail is not None and audio_tail is not None:
        args.update(latent_guide_args(kwargs, video_tail, audio_tail, timeline_guide=timeline_guide))
    return args


def seam_window_args(
    seam: SeamOptions,
    kwargs: Mapping[str, Any],
    window: ContinuationWindow,
    index: int,
    video_tail: torch.Tensor | None,
    audio_tail: torch.Tensor | None,
) -> dict[str, Any]:
    args: dict[str, Any] = {}
    if seam.seed_stride:
        args["seed"] = kwargs["seed"] + index * seam.seed_stride
    if seam.clock == "window":
        args.update(temporal_offset=0.0, media_time_origin=None)
    if seam.pin != "none" and video_tail is not None and audio_tail is not None:
        chunk = seam.pin == "chunk"
        count = _CHUNK_LATENTS if chunk else int(video_tail.shape[2])
        audio_count = int(audio_tail.shape[-1])
        if chunk:
            audio_count = _audio_t(window.start + _CHUNK_FRAMES) - window.audio_start
        video = minimax_h3_patchify_video_latent(video_tail[:, :, :count], patch_size=(1, 2, 2))
        audio = minimax_h3_pack_audio_latent(audio_tail[..., :audio_count])
        steps = torch.arange(audio_count, device=audio.device)
        args["pins"] = (
            (torch.arange(video.shape[0], device=video.device), video, count),
            (torch.cat((steps, steps + window.audio_end - window.audio_start)), audio, 1),
        )
    return args


def _seam_record(
    seam: SeamOptions | None, args: Mapping[str, Any], window: ContinuationWindow, index: int, count: int
) -> dict[str, Any] | None:
    if seam is None or (torch.distributed.is_initialized() and torch.distributed.get_rank() != 0):
        return None
    if torch.accelerator.is_available():
        torch.accelerator.reset_peak_memory_stats()
    return {
        "index": index,
        "windows": count,
        "start": window.start,
        "end": window.end,
        "overlap": window.overlap,
        "seed": args.get("seed"),
        "temporal_offset": args["temporal_offset"],
        "media_time_origin": args.get("media_time_origin"),
        "pin": seam.pin,
        "clock": seam.clock,
        "lookahead": seam.lookahead,
    }


def _seam_clock() -> float:
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()
    return time.perf_counter()


def _seam_memory() -> dict[str, float]:
    if not torch.accelerator.is_available():
        return {}
    return {
        "peak_alloc_gib": round(torch.accelerator.max_memory_allocated() / 2**30, 3),
        "peak_reserved_gib": round(torch.accelerator.max_memory_reserved() / 2**30, 3),
    }


def _rounded(values: torch.Tensor) -> list[float]:
    return [round(float(value), 5) for value in values.tolist()]


def _seam_adherence(
    sampled_video: torch.Tensor, sampled_audio: torch.Tensor, video_tail: torch.Tensor, audio_tail: torch.Tensor
) -> dict[str, Any]:
    count = int(video_tail.shape[2])
    tail = video_tail.float().transpose(0, 2).reshape(count, -1)
    new = sampled_video[:, :, :count].to(tail.device).float().transpose(0, 2).reshape(count, -1)
    scale = tail.norm(dim=1).clamp_min(1e-12)
    audio = audio_tail.float()
    audio_new = sampled_audio[..., : audio.shape[-1]].to(audio.device).float()
    audio_scale = audio.norm().clamp_min(1e-12)
    return {
        "video": _rounded((new - tail).norm(dim=1) / scale),
        "video_motion": _rounded((tail[1:] - tail[:-1]).norm(dim=1) / scale[:-1]),
        "audio": round(float((audio_new - audio).norm() / audio_scale), 5),
        "audio_motion": round(float((audio[..., 1:] - audio[..., :-1]).norm() / audio_scale), 5),
    }


def diffuse_continuation(
    diffuse: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    kwargs: dict[str, Any],
    *,
    window_frames: int,
    overlap_frames: int,
    text_conditioning: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
    prefix: LatentPrefix | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Denoise fresh windows; retain old latents and append only new suffixes.

    Guides are extra condition rows sharing the new target's temporal origin,
    not a masked target prefix. Audio boundaries refer to the cumulative frame
    timeline, preventing per-window rounding from accumulating A/V drift.
    Each window shifts temporal media positions onto that same global timeline
    before RoPE is evaluated; text and static image references remain fixed.

    A ``prefix`` seeds the cumulative latents with an earlier clip's tail: the
    first window repeats it behind a hidden overlap, and the result starts with
    that tail, so callers trim ``prefix.overlap_frames`` after decoding.
    """
    seam = kwargs.get("seam")
    lookahead = seam.lookahead if seam is not None else 0
    origin = prefix.start_frame if prefix is not None else 0
    windows = plan_continuation_windows(
        kwargs["num_frames"],
        window_frames,
        overlap_frames,
        first_overlap=prefix.overlap_frames if prefix is not None else 0,
        origin=origin,
        lookahead=lookahead,
    )
    if text_conditioning is not None and len(text_conditioning) != len(windows):
        raise OmniClientError("MiniMax H3 requires one text conditioning per continuation window")
    # Different prompts have different prefix lengths. Keep the media clock
    # anchored after the longest prefix so text length cannot shift AV time.
    media_time_origin = max(item[0].shape[0] for item in text_conditioning) if text_conditioning else None
    video, audio = (prefix.video, prefix.audio) if prefix is not None else (None, None)
    if prefix is not None and seam is not None and seam.pin != "none" and not lookahead:
        end = prefix.start_frame + prefix.overlap_frames
        if _audio_t(end) * 3 > end * 5:
            audio = audio[..., :-1]
    for index, window in enumerate(windows):
        if index and lookahead:
            video = video[:, :, : _video_t(window.start + window.overlap - origin)]
            audio = audio[..., : _audio_t(window.start + window.overlap) - _audio_t(origin)]
        overlap_v = _video_t(window.overlap) if window.overlap else 0
        guided = video is not None and audio is not None
        video_tail = video[:, :, -overlap_v:] if guided else None
        audio_tail = audio[..., -window.overlap_audio_t :] if guided else None
        guide = seam is None or seam.pin != "only"
        args = window_args(
            kwargs,
            window,
            video_tail=video_tail if guide else None,
            audio_tail=audio_tail if guide else None,
            timeline_guide=prefix is not None and index == 0,
            audio_origin=_audio_t(origin),
        )
        if text_conditioning is not None:
            args["text_embeddings"], args["text_tags"] = text_conditioning[index]
            args["media_time_origin"] = media_time_origin
        if seam is not None:
            args.update(seam_window_args(seam, kwargs, window, index, video_tail, audio_tail))
        logger.info(
            "MiniMax H3 continuation window %d/%d: frames [%d, %d), hidden overlap %d, temporal offset %.6f",
            index + 1,
            len(windows),
            window.start,
            window.end,
            window.overlap,
            args["temporal_offset"],
        )
        record = _seam_record(seam, args, window, index, len(windows))
        if record is not None and seam.diag and "pins" in args:
            args["pin_trace"] = record["pin_trace"] = []
        started = _seam_clock() if record is not None else 0.0
        sampled_video, sampled_audio = diffuse(**args)
        if record is not None:
            record["wall_s"] = round(_seam_clock() - started, 3)
            record.update(_seam_memory())
            if seam.diag and guided:
                record["adherence"] = _seam_adherence(sampled_video, sampled_audio, video_tail, audio_tail)
            logger.info("MiniMax H3 seam %s", json.dumps(record, separators=(",", ":")))
        if video is None or audio is None:
            video, audio = sampled_video, sampled_audio
        else:
            tail_a = int(audio_tail.shape[-1])
            keep_v, keep_a = overlap_v, tail_a
            if seam is not None and seam.pin == "chunk":
                keep_v, keep_a = _CHUNK_LATENTS, _audio_t(window.start + _CHUNK_FRAMES) - window.audio_start
                video = video[:, :, : video.shape[2] - overlap_v + keep_v]
                audio = audio[..., : audio.shape[-1] - tail_a + keep_a]
            video = torch.cat((video, sampled_video[:, :, keep_v:].to(video.device)), dim=2)
            audio = torch.cat((audio, sampled_audio[..., keep_a:].to(audio.device)), dim=-1)
    assert video is not None and audio is not None
    if video.shape[2] != kwargs["latent_t"] or audio.shape[-1] != kwargs["audio_t"]:
        raise RuntimeError("MiniMax H3 continuation produced inconsistent cumulative AV lengths")
    return video, audio

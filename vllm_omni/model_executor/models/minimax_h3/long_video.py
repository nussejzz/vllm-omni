# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared long-video request limits for encoder and diffusion entry points."""

import math
import os
from collections.abc import Mapping
from typing import Any

from vllm_omni.errors import OmniClientError

MINIMAX_H3_MAX_FULL_SECONDS = 30.0
MINIMAX_H3_MAX_FULL_SECONDS_ENV = "VLLM_OMNI_H3_MAX_FULL_SECONDS"
MINIMAX_H3_MAX_CONTINUATION_SECONDS = 300.0
MINIMAX_H3_MAX_REFERENCE_SECONDS = 15.0


def max_full_seconds() -> float:
    raw = os.environ.get(MINIMAX_H3_MAX_FULL_SECONDS_ENV, "").strip()
    if not raw:
        return MINIMAX_H3_MAX_FULL_SECONDS
    try:
        seconds = float(raw)
    except ValueError:
        seconds = math.nan
    if not MINIMAX_H3_MAX_FULL_SECONDS <= seconds <= 60.0:
        raise ValueError(f"{MINIMAX_H3_MAX_FULL_SECONDS_ENV} must be in [30, 60] seconds, got {raw!r}")
    return seconds


max_full_seconds()


def resolve_long_video_mode(extra: Mapping[str, Any], task: str) -> str:
    long_video = extra.get("long_video", False)
    if not isinstance(long_video, bool):
        raise OmniClientError("MiniMax H3 long_video must be a boolean")
    mode = extra.get("long_video_mode", "continuation" if long_video and task == "ref2va" else "full")
    if mode not in ("full", "continuation"):
        raise OmniClientError("MiniMax H3 long_video_mode must be full or continuation")
    if mode == "continuation" and task != "ref2va":
        raise OmniClientError("MiniMax H3 continuation requires Ref2VA request execution (not step execution)")
    return mode


def max_output_seconds(extra: Mapping[str, Any], task: str) -> float:
    mode = resolve_long_video_mode(extra, task)
    if not extra.get("long_video", False):
        return 15.0
    return MINIMAX_H3_MAX_CONTINUATION_SECONDS if mode == "continuation" else max_full_seconds()


def _aligned_frame_limit(seconds: float) -> int:
    requested_max = int(seconds * 24)
    return ((requested_max - 5 + 16) // 17) * 17 + 5


def max_reference_seconds(num_frames: int) -> float:
    if num_frames > _aligned_frame_limit(MINIMAX_H3_MAX_REFERENCE_SECONDS):
        return max_full_seconds()
    return MINIMAX_H3_MAX_REFERENCE_SECONDS


def max_request_reference_seconds(extra: Mapping[str, Any], task: str, num_frames: int) -> float:
    if not extra.get("long_video", False) or resolve_long_video_mode(extra, task) != "full":
        return MINIMAX_H3_MAX_REFERENCE_SECONDS
    return max_reference_seconds(num_frames)


def validate_encoded_frame_limit(extra: Mapping[str, Any], task: str, num_frames: int) -> None:
    """Apply the same cap to externally encoded requests, allowing grid rounding."""
    seconds = max_output_seconds(extra, task)
    aligned_max = _aligned_frame_limit(seconds)
    if num_frames > aligned_max:
        raise OmniClientError(
            f"MiniMax H3 output duration exceeds the {seconds:g}-second request limit "
            f"({aligned_max} frames after alignment); got {num_frames} frames"
        )

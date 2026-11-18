# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bit-exact fused SiLU and causal-convolution padding for the H3 VAE encoder.

The checkpoint's encoder prepares every padded 3D convolution input with a
chain of full-size copies: ``_split_time_from_batch`` (BTCHW -> BCTHW
``contiguous``), ``F.silu``, a spatial ``reflect`` ``F.pad`` and a causal
constant ``F.pad`` over time. This kernel reads the source once through its
strides and writes the contiguous padded tensor the convolution receives:
reflected rows and columns, zero leading frames and, optionally, SiLU.

SiLU keeps the eager CUDA formula ``x / (1 + exp(-x))`` in FP32 with the
libdevice ``expf`` and an IEEE ``div.rn`` division; Triton's default
approximate exponential and division change the encoder latents. All 2**32
FP32 inputs match ``F.silu`` bit for bit on SM90. Floating-point contraction
is disabled at launch.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:
    from triton.language.extra import libdevice

    @triton.jit
    def _reflect_index(index, size):
        index = tl.where(index < 0, -index, index)
        return tl.where(index >= size, 2 * size - 2 - index, index)

    @triton.jit
    def _causal_reflect_pad_kernel(
        output_ptr,
        source_ptr,
        channels,
        height,
        width,
        source_stride_b,
        source_stride_c,
        source_stride_t,
        source_stride_h,
        source_stride_w,
        output_frames,
        output_height,
        output_width,
        front,
        top,
        left,
        right,
        column_blocks,
        plane_tiles,
        apply_silu: tl.constexpr,
        block_h: tl.constexpr,
        block_w: tl.constexpr,
        border_w: tl.constexpr,
    ):
        # Each program covers one tile of rows and source columns of one
        # output (batch, channel, frame) plane, so the index decomposition
        # stays scalar per program.
        program = tl.program_id(0)
        plane = program // plane_tiles
        tile = program - plane * plane_tiles
        row_block = tile // column_blocks
        column_block = tile - row_block * column_blocks
        frame = plane % output_frames - front
        batch_channel = plane // output_frames
        live = frame >= 0
        source_plane = (
            (batch_channel // channels).to(tl.int64) * source_stride_b
            + (batch_channel % channels).to(tl.int64) * source_stride_c
            + tl.maximum(frame, 0).to(tl.int64) * source_stride_t
        )

        output_rows = row_block * block_h + tl.arange(0, block_h)
        row_mask = output_rows < output_height
        source_rows = source_plane + _reflect_index(output_rows - top, height).to(tl.int64) * source_stride_h
        output_rows = plane.to(tl.int64) * output_height * output_width + output_rows.to(tl.int64) * output_width

        # Interior: source columns are read contiguously and stored shifted
        # by ``left``. Leading causal frames store +0.0 like the constant pad.
        columns = column_block * block_w + tl.arange(0, block_w)
        mask = row_mask[:, None] & (columns < width)[None, :]
        load_mask = mask & live
        values = tl.load(
            source_ptr + source_rows[:, None] + columns[None, :] * source_stride_w,
            mask=load_mask,
            other=0.0,
        )
        if apply_silu:
            values = tl.div_rn(values, 1.0 + libdevice.exp(-values))
        tl.store(
            output_ptr + output_rows[:, None] + left + columns[None, :],
            tl.where(load_mask, values, 0.0),
            mask=mask,
        )

        if column_block == 0:
            # Reflected border columns: [0, left) and [left + width, output_width).
            border = tl.arange(0, border_w)
            output_columns = tl.where(border < left, border, width + border)
            source_columns = _reflect_index(output_columns - left, width)
            border_mask = row_mask[:, None] & (border < left + right)[None, :]
            border_load_mask = border_mask & live
            border_values = tl.load(
                source_ptr + source_rows[:, None] + source_columns[None, :] * source_stride_w,
                mask=border_load_mask,
                other=0.0,
            )
            if apply_silu:
                border_values = tl.div_rn(border_values, 1.0 + libdevice.exp(-border_values))
            tl.store(
                output_ptr + output_rows[:, None] + output_columns[None, :],
                tl.where(border_load_mask, border_values, 0.0),
                mask=border_mask,
            )


# Warmed SM90 sweep over the encoder shapes: ~1024 elements per program.
_ELEMENTS_PER_PROGRAM = 1024
_MAX_BLOCK_W = 256


def try_causal_reflect_pad_exact(
    source: torch.Tensor,
    *,
    front: int,
    top: int,
    bottom: int,
    left: int,
    right: int,
    apply_silu: bool,
) -> torch.Tensor | None:
    """Pad a BCTHW-shaped ``source`` view, or return ``None`` outside the contract.

    The result equals ``F.pad(F.pad(act(source), (left, right, top, bottom,
    0, 0), mode="reflect"), (0, 0, 0, 0, front, 0))`` bit for bit, where
    ``act`` is ``F.silu`` when ``apply_silu`` is set and the identity
    otherwise. ``source`` may have any strides (for example the BTCHW
    GroupNorm output viewed as BCTHW); the result is a fresh contiguous FP32
    tensor.
    """

    pads = (front, top, bottom, left, right)
    if not (
        HAS_TRITON
        and torch.version.hip is None
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and source.is_cuda
        # Triton launches on the current device; a tensor on another device
        # keeps the reference path.
        and source.device.index == torch.accelerator.current_device_index()
        and source.dtype == torch.float32
        and source.ndim == 5
        and source.numel() > 0
        and all(type(pad) is int and pad >= 0 for pad in pads)
    ):
        return None
    batch, channels, frames, height, width = source.shape
    # ``reflect`` requires each pad to be smaller than its dimension; the
    # checkpoint path raises there, so keep that error on the reference path.
    if max(top, bottom) >= height or max(left, right) >= width:
        return None
    # Column offsets are formed in 32-bit lanes before the 64-bit row base.
    if (width - 1) * source.stride(4) >= 2**31:
        return None

    output_frames = frames + front
    output_height = height + top + bottom
    output_width = width + left + right
    planes = batch * channels * output_frames
    block_w = min(_MAX_BLOCK_W, triton.next_power_of_2(width))
    block_h = min(triton.next_power_of_2(output_height), max(1, _ELEMENTS_PER_PROGRAM // block_w))
    column_blocks = triton.cdiv(width, block_w)
    plane_tiles = triton.cdiv(output_height, block_h) * column_blocks
    if planes * plane_tiles >= 2**31:
        return None

    output = torch.empty(
        (batch, channels, output_frames, output_height, output_width),
        dtype=source.dtype,
        device=source.device,
    )
    _causal_reflect_pad_kernel[(planes * plane_tiles,)](
        output,
        source,
        channels,
        height,
        width,
        *source.stride(),
        output_frames,
        output_height,
        output_width,
        front,
        top,
        left,
        right,
        column_blocks,
        plane_tiles,
        apply_silu=apply_silu,
        block_h=block_h,
        block_w=block_w,
        border_w=max(2, triton.next_power_of_2(left + right)),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


__all__ = ["try_causal_reflect_pad_exact"]

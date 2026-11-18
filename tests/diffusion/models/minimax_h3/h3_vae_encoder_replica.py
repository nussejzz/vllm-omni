# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-process replica of the MiniMax H3 checkpoint's video VAE encoder.

Mirrors ``conv.py``, ``norm.py`` and ``vae_cnn.py`` shipped with the
checkpoint (class names, attributes and forward op order) without the
spatial-parallel collectives, so the encoder fusion can be checked against the
checkpoint forward without downloading remote code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, padding_mode="zeros", causal=True):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
        )
        self.pad_mode = "constant" if padding_mode == "zeros" else padding_mode
        self.pad_mode_t = "constant" if causal else "replicate"
        self.causal = causal

    def _apply_temporal_padding(self, x):
        if x.shape[2] > 1:
            front = self.padding[0] * 2 if self.causal else self.padding[0]
            back = 0 if self.causal else self.padding[0]
            return F.pad(x, (0, 0, 0, 0, front, back), mode=self.pad_mode_t)
        if self.pad_mode_t == "constant":
            zeros = torch.zeros_like(x[:, :, :1, :, :]).expand(-1, -1, self.kernel_size[0] - 1, -1, -1)
            return torch.cat([zeros, x], dim=2)
        return x.expand(-1, -1, self.kernel_size[0], -1, -1)

    def _apply_padding(self, x):
        if sum(self.padding) == 0:
            return x
        x = F.pad(x, (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 0, 0), mode=self.pad_mode)
        return self._apply_temporal_padding(x)

    def forward(self, x):
        if sum(self.padding) == 0:
            return super().forward(x)
        x = self._apply_padding(x)
        return F.conv3d(x, self.weight, self.bias, stride=self.stride, padding=0, dilation=self.dilation)


class SpatialParallelConv3d(BaseConv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_parallel = False
        self.chunk_dim = -1


def _merge_time_to_batch(x):
    batch, channels, depth, height, width = x.shape
    return x.permute(0, 2, 1, 3, 4).contiguous().view(batch * depth, channels, 1, height, width)


def _split_time_from_batch(x, batch):
    batch_depth, channels, _, height, width = x.shape
    return x.view(batch, batch_depth // batch, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()


class SpatialParallelGroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_parallel = False

    def forward(self, input):
        if not self.spatial_parallel:
            return nn.GroupNorm.forward(self, input)
        raise NotImplementedError("spatial parallel GroupNorm is not replicated")


class TemporalIsolatedSpatialParallelGroupNorm(SpatialParallelGroupNorm):
    def forward(self, input):
        if input.dim() == 5:
            batch = input.shape[0]
            return _split_time_from_batch(super().forward(_merge_time_to_batch(input)), batch)
        return super().forward(input)


def _group_norm(channels, time_isolated):
    norm_cls = TemporalIsolatedSpatialParallelGroupNorm if time_isolated else SpatialParallelGroupNorm
    norm = norm_cls(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
    with torch.no_grad():
        norm.weight.normal_(1.0, 0.2)
        norm.bias.normal_(0.0, 0.2)
    return norm


class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_stride=1, space_stride=2, padding_mode="reflect"):
        super().__init__()
        self.time_stride = time_stride
        self.space_stride = space_stride
        self.conv = SpatialParallelConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=(1, 0, 0),
            stride=(time_stride, space_stride, space_stride),
            padding_mode=padding_mode,
        )
        self.causal = self.conv.causal
        self.pad_mode = self.conv.pad_mode

    def forward(self, x):
        if self.space_stride == 2:
            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode=self.pad_mode)
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode="reflect", time_isolated=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_fused_norm = False
        self.norm1 = _group_norm(in_channels, time_isolated)
        self.norm2 = _group_norm(out_channels, time_isolated)
        self.conv1 = SpatialParallelConv3d(
            in_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode
        )
        self.conv2 = SpatialParallelConv3d(
            out_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode
        )
        if in_channels != out_channels:
            self.nin_shortcut = SpatialParallelConv3d(
                in_channels, out_channels, kernel_size=1, padding_mode=padding_mode
            )

    def forward(self, x, zq=None):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class EncoderFCN3D(nn.Module):
    def __init__(
        self, ch_mult=(1, 2, 2), space_down=(2, 2, 1), time_down=(1, 2, 1), padding_mode="reflect", time_isolated=True
    ):
        super().__init__()
        ch = 32
        self.num_levels = len(ch_mult)
        self.num_res_blocks = [2] * self.num_levels
        self.use_fused_norm = False
        block_mid = [ch * mult for mult in ch_mult]
        block_in = [block_mid[0]] + block_mid[:-1]
        self.conv_in = SpatialParallelConv3d(3, block_in[0], kernel_size=3, padding=1, padding_mode=padding_mode)
        self.down = nn.ModuleList()
        for i_level in range(self.num_levels):
            down = nn.Module()
            down.block = nn.ModuleList(
                ResnetBlock3D(
                    block_in[i_level] if i == 0 else block_mid[i_level],
                    block_mid[i_level],
                    padding_mode=padding_mode,
                    time_isolated=time_isolated,
                )
                for i in range(self.num_res_blocks[i_level])
            )
            if space_down[i_level] * time_down[i_level] > 1:
                down.downsample = Downsample3D(
                    block_mid[i_level],
                    block_mid[i_level],
                    time_stride=time_down[i_level],
                    space_stride=space_down[i_level],
                    padding_mode=padding_mode,
                )
            self.down.append(down)
        self.norm_out = _group_norm(block_mid[-1], time_isolated)
        self.conv_out = SpatialParallelConv3d(block_mid[-1], 8, kernel_size=3, padding=1, padding_mode=padding_mode)

    def forward(self, x, zq=None):
        h = self.conv_in(x)
        for i_level in range(self.num_levels):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](h, zq)
            if hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample(h)
        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


def make_encoder(seed=0, **kwargs):
    torch.manual_seed(seed)
    return EncoderFCN3D(**kwargs).eval()

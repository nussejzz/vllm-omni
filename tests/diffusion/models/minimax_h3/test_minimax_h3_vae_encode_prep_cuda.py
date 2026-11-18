# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Device-resident reference frames normalize exactly like the host upload.

DiT ranks 1-7 keep the broadcast uint8 reference video on the device
(``VLLM_OMNI_H3_RANK_GPU_REFERENCE_FRAMES``); the video VAE then slices and
pads it there. The normalized FP32 tensor must equal the host path's bit for
bit, because it is the only input to the checkpoint's encoder.
"""

import numpy as np
import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


class _Normalize:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)


class _Processor:
    def __init__(self, pad):
        self.transform = _Normalize()
        self._pad = pad

    def align_video_length(self, video_length, mode="pad", granularity="chunk"):
        return self._pad


class _Model:
    clip_length = 4
    isolated_first_frame = False
    frame_pre_padding = 0

    def __init__(self, pad):
        self.processor = _Processor(pad)


@pytest.mark.parametrize(("num_frames", "pad"), [(10, 2), (8, 0), (2, 2), (10, 5)])
def test_resident_frames_normalize_bitwise_like_host_upload(num_frames, pad):
    vae = object.__new__(MiniMaxH3VideoVAE)
    vae.model = _Model(pad)
    device = torch.device("cuda", torch.accelerator.current_device_index())
    frames = np.random.default_rng(num_frames + pad).integers(0, 256, size=(num_frames, 18, 30, 3), dtype=np.uint8)
    expected = vae._stream_prepare_video_tensor(frames, device)
    resident = torch.from_numpy(frames).to(device)
    out = vae._stream_prepare_video_tensor(resident, device)
    assert out.device == device and out.shape == (3, num_frames + pad, 18, 30)
    assert torch.equal(out, expected)
    assert torch.equal(resident.cpu(), torch.from_numpy(frames))

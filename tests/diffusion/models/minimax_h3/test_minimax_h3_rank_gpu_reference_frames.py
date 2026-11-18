# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""DiT ranks 1-7 keep broadcast reference videos on the device.

``_distribute_media_inputs`` used to copy every broadcast tensor back to the
host; the reference video alone is a 384 MB pageable copy on the official
Ref2VA case, re-uploaded by the video VAE right after. Videos now stay where
the broadcast wrote them unless ``VLLM_OMNI_H3_RANK_GPU_REFERENCE_FRAMES=0``.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline_module
from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderMediaInput
from vllm_omni.model_executor.models.minimax_h3.encoder_processing import encode_media

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

ENV = pipeline_module.MINIMAX_H3_RANK_GPU_REFERENCE_FRAMES_ENV


def _media():
    return MiniMaxH3EncoderMediaInput(
        task="ref2va",
        height=32,
        width=32,
        num_frames=107,
        latent_t=32,
        audio_t=178,
        images=(torch.full((32, 32, 3), 7, dtype=torch.uint8),),
        videos=(
            torch.arange(4 * 32 * 32 * 3, dtype=torch.int64).remainder(256).to(torch.uint8).view(4, 32, 32, 3),
            torch.full((3, 32, 32, 3), 9, dtype=torch.uint8),
        ),
        video_audios=((torch.ones(48000), 16000), None),
        audios=((torch.zeros(64000), 32000),),
    )


def _distribute_on_rank(monkeypatch, rank, media):
    """Run ``_distribute_media_inputs`` as ``rank`` of two; report host copies."""
    tensors = media.to_mm_tensors()
    header = (media.to_metadata(), [value.dtype for value in tensors])
    monkeypatch.setattr(pipeline_module, "_dit_rank_world", lambda: (None, rank, 2))
    monkeypatch.setattr(
        pipeline_module.dist, "broadcast_object_list", lambda values, src, group: values.__setitem__(0, header)
    )
    broadcast = []

    def fake_broadcast(tensor, *, dtype, device):
        broadcast.append(tensors[len(broadcast)].clone())
        return broadcast[-1]

    monkeypatch.setattr(pipeline_module, "_broadcast_tensor", fake_broadcast)
    copied = []
    host_copy = torch.Tensor.cpu

    def recording_cpu(self, *args, **kwargs):
        copied.append(self)
        return host_copy(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", recording_cpu)
    model = SimpleNamespace(device=torch.device("cpu"))
    result = pipeline_module.MiniMaxH3Pipeline._distribute_media_inputs(model, media if rank == 0 else None)
    monkeypatch.setattr(torch.Tensor, "cpu", host_copy)
    copied_flags = [any(value is item for item in copied) for value in broadcast]
    return result, copied_flags


@pytest.mark.parametrize("value", [None, "", "1", "true", "ON"])
def test_nonzero_ranks_keep_reference_videos_resident(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    media = _media()
    result, copied = _distribute_on_rank(monkeypatch, 1, media)
    # to_mm_tensors order: image, video, video, embedded audio, standalone audio.
    assert copied == [True, False, False, True, True]
    assert all(torch.equal(a, b) for a, b in zip(result.to_mm_tensors(), media.to_mm_tensors(), strict=True))
    assert result.to_metadata() == media.to_metadata()


def test_non_uint8_videos_keep_the_host_copy(monkeypatch):
    # encode_media casts videos to uint8; that cast stays on the host for any
    # other dtype so it cannot differ from the previous path.
    monkeypatch.delenv(ENV, raising=False)
    media = _media()
    media = replace(media, videos=(media.videos[0].to(torch.float32), media.videos[1]))
    result, copied = _distribute_on_rank(monkeypatch, 1, media)
    assert copied == [True, True, False, True, True]
    assert result.videos[0].dtype == torch.float32
    assert all(torch.equal(a, b) for a, b in zip(result.to_mm_tensors(), media.to_mm_tensors(), strict=True))


@pytest.mark.parametrize("value", ["0", "false", " off "])
def test_switch_off_restores_host_copies(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    media = _media()
    result, copied = _distribute_on_rank(monkeypatch, 1, media)
    assert copied == [True] * 5
    assert all(torch.equal(a, b) for a, b in zip(result.to_mm_tensors(), media.to_mm_tensors(), strict=True))


def test_rank0_returns_its_own_media(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    media = _media()
    result, copied = _distribute_on_rank(monkeypatch, 0, media)
    assert result is media
    assert copied == [False] * 5


def test_invalid_switch_value_is_rejected(monkeypatch):
    monkeypatch.setenv(ENV, "2")
    with pytest.raises(ValueError, match=ENV):
        pipeline_module.rank_gpu_reference_frames()


class _RecordingVideoVAE:
    def __init__(self):
        self.received = []

    def encode_video(self, frames):
        self.received.append(frames)
        return torch.zeros(4, 96), (1, 4, 4)


def test_encode_media_hands_device_videos_to_the_vae_without_a_host_copy():
    resident = torch.empty(4, 32, 32, 3, dtype=torch.uint8, device="meta")
    media = MiniMaxH3EncoderMediaInput(
        task="ref2va",
        height=32,
        width=32,
        num_frames=107,
        latent_t=32,
        audio_t=178,
        videos=(resident, torch.full((3, 32, 32, 3), 9, dtype=torch.uint8)),
        video_audios=(None, None),
    )
    video_vae = _RecordingVideoVAE()
    assert encode_media(media, video_vae=video_vae, audio_vae=None, emit_conditioning=False) is None
    device_frames, host_frames = video_vae.received
    assert isinstance(device_frames, torch.Tensor) and device_frames.device.type == "meta"
    assert device_frames.dtype == torch.uint8 and device_frames.shape == resident.shape
    assert isinstance(host_frames, np.ndarray) and host_frames.dtype == np.uint8
    assert np.array_equal(host_frames, np.full((3, 32, 32, 3), 9, dtype=np.uint8))

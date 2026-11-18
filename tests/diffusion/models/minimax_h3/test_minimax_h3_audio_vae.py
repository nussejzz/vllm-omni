# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch
import torch.nn as nn

import vllm_omni.diffusion.models.minimax_h3.vae as h3_vae
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3AudioVAE, _AudioVAEDeterminismContext

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _graph_executor_optimize() -> bool:
    return torch._C._get_graph_executor_optimize()


class _RecordingDecoder:
    def __init__(self) -> None:
        self.optimize_during_decode: list[bool] = []

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.optimize_during_decode.append(_graph_executor_optimize())
        return torch.zeros(latent.shape[0], 1, 8)


def test_audio_vae_decode_runs_torchscript_without_executor_optimization():
    vae = MiniMaxH3AudioVAE.__new__(MiniMaxH3AudioVAE)
    nn.Module.__init__(vae)
    vae.config_dict = {"latent_channels": 2, "latents_mean": [0.0, 0.0], "latents_std": [1.0, 1.0]}
    vae.remote = _RecordingDecoder()
    before = _graph_executor_optimize()

    for _ in range(2):
        waveform = vae.decode_latent(torch.zeros(2, 2, 4))

    assert vae.remote.optimize_during_decode == [False, False]
    assert _graph_executor_optimize() == before
    assert tuple(waveform.shape) == (1, 2, 8)


def test_audio_vae_determinism_context_disables_and_restores_executor_optimization():
    before = _graph_executor_optimize()

    with _AudioVAEDeterminismContext():
        assert _graph_executor_optimize() is False

    assert _graph_executor_optimize() == before


class UpSample1d(nn.Module):
    """Stand-in with the remote BigVGAN ``UpSample1d`` contract and forward."""

    def __init__(self, ratio: int = 2, kernel_size: int = 12) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2
        self.register_buffer("filter", torch.randn(1, 1, kernel_size, generator=torch.Generator().manual_seed(7)))

    def forward(self, x):
        _, channels, _ = x.shape
        x = nn.functional.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * nn.functional.conv_transpose1d(
            x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels
        )
        return x[..., self.pad_left : -self.pad_right]


@pytest.mark.parametrize(("ratio", "kernel_size"), [(2, 12), (3, 12), (2, 4)])
@pytest.mark.parametrize(("channels", "length"), [(1, 5), (8, 97), (16, 300)])
def test_polyphase_upsample_matches_transposed_conv(ratio, kernel_size, channels, length):
    module = UpSample1d(ratio, kernel_size).double()
    x = torch.randn(2, channels, length, dtype=torch.float64, generator=torch.Generator().manual_seed(length))

    expected = module.forward(x)
    actual = h3_vae._polyphase_upsample1d_forward(module, x)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_install_switches_only_supported_upsamplers():
    supported = UpSample1d()
    odd = UpSample1d()
    odd.stride = 1
    remote = nn.Sequential(supported, nn.Sequential(odd), nn.Conv1d(2, 2, 3))
    x = torch.randn(1, 4, 50)
    before = supported(x)

    assert h3_vae._install_deterministic_audio_upsample(remote) == 1
    assert "forward" in supported.__dict__
    assert "forward" not in odd.__dict__
    torch.testing.assert_close(supported(x), before, rtol=1e-5, atol=1e-6)


def test_install_keeps_remote_forward_when_self_check_fails(monkeypatch):
    module = UpSample1d()
    monkeypatch.setattr(h3_vae, "_polyphase_upsample1d_forward", lambda self, x: UpSample1d.forward(self, x) + 1.0)

    assert h3_vae._install_deterministic_audio_upsample(nn.Sequential(module)) == 0
    assert "forward" not in module.__dict__


@pytest.mark.parametrize(("raw", "expected"), [(None, True), ("", True), ("1", True), (" 1 ", True), ("0", False)])
def test_deterministic_audio_switch(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(h3_vae.MINIMAX_H3_DETERMINISTIC_AUDIO_ENV, raising=False)
    else:
        monkeypatch.setenv(h3_vae.MINIMAX_H3_DETERMINISTIC_AUDIO_ENV, raw)
    assert h3_vae._deterministic_audio_enabled() is expected


def test_deterministic_audio_switch_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv(h3_vae.MINIMAX_H3_DETERMINISTIC_AUDIO_ENV, "true")
    with pytest.raises(ValueError, match="must be '1' or '0'"):
        h3_vae._deterministic_audio_enabled()


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_deterministic_decode_context_scopes_cudnn(monkeypatch, initial, fail):
    monkeypatch.setattr(h3_vae.current_omni_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", initial)
    others = {name: getattr(torch.backends.cudnn, name) for name in ("enabled", "benchmark", "allow_tf32")}
    try:
        with h3_vae._deterministic_audio_decode_context(torch.device("cuda", 0)):
            assert torch.backends.cudnn.deterministic
            assert {name: getattr(torch.backends.cudnn, name) for name in others} == others
            if fail:
                raise RuntimeError("injected")
    except RuntimeError as exc:
        assert fail and str(exc) == "injected"
    assert torch.backends.cudnn.deterministic is initial

    with h3_vae._deterministic_audio_decode_context(torch.device("cpu")):
        assert torch.backends.cudnn.deterministic is initial


class _FlagRecordingDecoder:
    def __init__(self) -> None:
        self.deterministic: list[bool] = []

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.deterministic.append(torch.backends.cudnn.deterministic)
        return torch.zeros(latent.shape[0], 1, 8)


@pytest.mark.parametrize("enabled", [False, True])
def test_audio_vae_decode_uses_switch(monkeypatch, enabled):
    seen = []

    def fake_context(device):
        seen.append(device)
        return h3_vae.nullcontext()

    monkeypatch.setattr(h3_vae, "_deterministic_audio_decode_context", fake_context)
    vae = MiniMaxH3AudioVAE.__new__(MiniMaxH3AudioVAE)
    nn.Module.__init__(vae)
    vae.config_dict = {"latent_channels": 2, "latents_mean": [0.0, 0.0], "latents_std": [1.0, 1.0]}
    vae.remote = _FlagRecordingDecoder()
    vae.deterministic_decode = enabled

    vae.decode_latent(torch.zeros(2, 2, 4))

    assert seen == ([torch.device("cpu")] if enabled else [])


def test_install_self_check_ignores_default_device():
    # Model construction can run under a non-CPU default device (vLLM loads on the GPU).
    module = UpSample1d()
    with torch.device("meta"):
        assert h3_vae._install_deterministic_audio_upsample(nn.Sequential(module)) == 1
    assert "forward" in module.__dict__

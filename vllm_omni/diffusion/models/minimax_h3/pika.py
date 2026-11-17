# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pika H3 Flex release contract and request-owned inference noise.

Release: https://huggingface.co/Pika-Labs/Pika-H3-Flex (adapter revision 0ddf55a; the
ref2va ``adapter.safetensors`` digest is identical at 3ff1664).
Sampling / noise convention: Pika-Labs/Pika-H3-Flex at f961d7e (Apache-2.0),
``src/pika_h3/core/{sampling.py,wrapper.py,evaluation.py,schedules.py}``.
Ported from the Pika part of vllm-project/vllm-omni draft PR #9777 (head 72bc945)
onto the request-local sampler framework of ``vllm_omni.diffusion.sampler``.

Contract summary (``pika-flex-noise-v1``):

* six transformer forwards on knots ``[1000, 750, 500, 250, 100, 40, 0] / 1000``,
  shifted per modality (video 12, audio 3) in float64 and stored as float32;
* ``x0 = x_t + sigma_grid * v`` (the grid sigma, not ``1 - (1 - sigma)``);
* re-noise ``x_{i+1} = (1 - sigma_{i+1}) * x0 + sigma_{i+1} * eps`` in float32 with
  fresh noise per step, video drawn before audio; the terminal 0 knot draws nothing;
* three independent device generators per request: initial noise ``seed``,
  re-noise ``seed + 1`` and reference augmentation ``seed + 7919``, all drawn in
  Pika's external layouts (video ``[1, F, 24, H, W]``, audio ``[1, T, 64]``) and
  only then packed into rows.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch

from vllm_omni.diffusion.sampler import RenoiseNoiseSource

from .packed_tokens import minimax_h3_patchify_video_latent

PIKA_BASE_REPO = "MiniMaxAI/MiniMax-H3"
PIKA_BASE_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
PIKA_KNOTS = (1000, 750, 500, 250, 100, 40, 0)
PIKA_VIDEO_SHIFT = 12.0
PIKA_AUDIO_SHIFT = 3.0
PIKA_RANK = 128
PIKA_ALPHA = 128
PIKA_TARGET_MODULES = ("to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2")
PIKA_NOISE_VERSION = "pika-flex-noise-v1"
PIKA_RENOISE_SEED_OFFSET = 1
PIKA_AUGMENTATION_SEED_OFFSET = 7919
# Two requests whose seeds differ by one of these share a generator stream
# (seed S's re-noise stream is seed S+1's initial stream; its augmentation
# stream is seed S+7919's initial stream and seed S+7918's re-noise stream), so
# their samples are correlated. Batch generation should space seeds further
# apart, e.g. ``base + k * 10007``.
PIKA_CORRELATED_SEED_OFFSETS = (
    PIKA_RENOISE_SEED_OFFSET,
    PIKA_AUGMENTATION_SEED_OFFSET - PIKA_RENOISE_SEED_OFFSET,
    PIKA_AUGMENTATION_SEED_OFFSET,
)
# Visual reference rows are noised to t=0.999 (``H3DiffusionWrapper._noise_condition_rows``).
PIKA_CONDITION_NOISE_AUG = 0.999
PIKA_NUMERICS_POLICY = {"ref2va": "fp32_conditioning_v1", "fl2va": "bf16_conditioning_v1"}
# FL2VA (bf16 conditioning, crop-fit keyframes) is not ported yet.
PIKA_SERVED_TASKS = frozenset({"ref2va"})


def pika_sigmas(shift: float) -> list[float]:
    """Pika's shifted ladder: float64 ``s*q / (1 + (s-1)*q)`` stored as float32.

    Returned as Python floats that are exactly the float32 values, so ``1.0 - sigma``
    (the AdaLN time, written into a float32 tensor) and ``x + sigma * v`` see the
    same operands as Pika (``schedules.py:20-21,219-227``).
    """
    base = torch.tensor(PIKA_KNOTS, dtype=torch.float64) / 1000
    return (shift * base / (1 + (shift - 1) * base)).float().tolist()


@dataclass(frozen=True)
class PikaSpec:
    """The validated identity of one Pika adapter release."""

    task: str
    sha256: str
    release_id: str
    numerics_policy: str
    weights_path: str

    name: ClassVar[str] = "Pika H3 Flex"
    denoise_steps: ClassVar[int] = len(PIKA_KNOTS) - 1

    @property
    def conditioning_dtype(self) -> torch.dtype:
        """fp32_conditioning_v1 keeps visual condition latents FP32; bf16_conditioning_v1 rounds them."""
        return torch.float32 if self.numerics_policy == "fp32_conditioning_v1" else torch.bfloat16

    @property
    def base_schedule(self) -> tuple[float, ...]:
        return tuple(knot / 1000 for knot in PIKA_KNOTS)

    @property
    def video_sigmas(self) -> list[float]:
        return pika_sigmas(PIKA_VIDEO_SHIFT)

    @property
    def audio_sigmas(self) -> list[float]:
        return pika_sigmas(PIKA_AUDIO_SHIFT)

    def check_request(self, sampling: Any, task: str, *, step_execution: bool = False) -> None:
        """Reject request options that would leave Pika's sampling contract.

        Raises ``ValueError``; the pipeline turns it into a client error.
        """
        if task != self.task:
            raise ValueError(f"{self.name} {self.task} adapter cannot serve task={task}")
        steps = getattr(sampling, "num_inference_steps", None)
        if steps is not None and int(steps) != self.denoise_steps:
            raise ValueError(
                f"{self.name} requires num_inference_steps={self.denoise_steps} (transformer forwards) or omitted, "
                f"got {steps}"
            )
        if step_execution and steps is None:
            raise ValueError(f"{self.name} step execution requires explicit num_inference_steps={self.denoise_steps}")
        extra = getattr(sampling, "extra_args", None) or {}
        for key, value in (("flow_shift", PIKA_VIDEO_SHIFT), ("audio_flow_shift", PIKA_AUDIO_SHIFT)):
            raw = extra.get(key, value)
            try:
                ok = float(raw) == value
            except (TypeError, ValueError):
                ok = False
            if not ok:
                raise ValueError(f"{self.name} requires {key}={value:g} (or omitted), got {raw!r}")
        if extra.get("sampler") not in (None, "renoise"):
            raise ValueError(f"{self.name} uses its own re-noising sampler; remove sampler={extra['sampler']!r}")
        if extra.get("long_video_mode", "full") != "full" or extra.get("continuation_prompts") is not None:
            raise ValueError(f"{self.name} generates full clips; continuation is not supported")
        if extra.get("continue_from") is not None:
            raise ValueError(f"{self.name} does not support continue_from")
        if extra.get("audio_mode", "native") != "native":
            raise ValueError(f"{self.name} requires audio_mode=native")
        if extra.get("regenerate", False):
            raise ValueError(f"{self.name} does not support regenerate")
        if getattr(sampling, "quality", None) == "high":
            raise ValueError(f"{self.name} requires uncached denoising (Cache-DiT skips steps); use quality=lossless")
        if getattr(sampling, "lora_request", None) is not None:
            raise ValueError(f"{self.name} serves its startup adapter at scale 1; remove the request-level lora field")
        # One output per request. The pipeline seeds output k with ``seed + k``,
        # and under pika-flex-noise-v1 output k's initial-noise stream (seed + k)
        # is output k-1's re-noise stream (seed + k - 1 + 1): the samples would be
        # strongly correlated without any error. Send separate requests with
        # well-separated seeds instead (see PIKA_CORRELATED_SEED_OFFSETS).
        num_outputs = getattr(sampling, "num_outputs_per_prompt", None)
        if num_outputs is not None and (isinstance(num_outputs, bool) or num_outputs != 1):
            raise ValueError(
                f"{self.name} produces one output per request (num_outputs_per_prompt=1); "
                f"got {num_outputs!r}. Send one request per seed and do not use adjacent seeds"
            )


def _require(manifest: dict[str, Any], key: str, expected: Any) -> None:
    if manifest.get(key) != expected:
        raise ValueError(f"Unsupported Pika manifest {key}: expected {expected!r}, got {manifest.get(key)!r}")


def read_pika_spec(path: str | Path) -> PikaSpec | None:
    """Identify a Pika adapter by its sibling ``manifest.json``, never by a filename.

    ``path`` is the ``ref2va/`` (or ``fl2va/``) release directory or its
    ``adapter.safetensors``. Returns ``None`` for anything that is not a Pika
    release, so other ``--lora-path`` artifacts keep their own loaders.
    """
    path = Path(path)
    root = path if path.is_dir() else path.parent
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        if path.is_dir() and any((path / task / "manifest.json").is_file() for task in PIKA_NUMERICS_POLICY):
            raise ValueError("Select the Pika ref2va/ (or fl2va/) adapter directory, not the repository root")
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict) or manifest.get("model") != "pika-flex":
        return None
    task = manifest.get("task")
    if task not in PIKA_NUMERICS_POLICY:
        raise ValueError(f"Unsupported Pika task: {task!r}")
    # Same checks as Pika's own loader (src/pika_h3/pipeline.py:34-49).
    _require(manifest, "format_version", 1)
    _require(manifest, "release_id", f"pika-flex-{task}-v1")
    _require(manifest, "rank", PIKA_RANK)
    _require(manifest, "alpha", PIKA_ALPHA)
    _require(manifest, "trunk", "transformer_ref" if task == "ref2va" else "transformer")
    _require(manifest, "numerics_policy", PIKA_NUMERICS_POLICY[task])
    _require(manifest, "base", {"repo_id": PIKA_BASE_REPO, "revision": PIKA_BASE_REVISION})
    _require(
        manifest,
        "sampling",
        {
            "knots": list(PIKA_KNOTS),
            "video_shift": PIKA_VIDEO_SHIFT,
            "audio_shift": PIKA_AUDIO_SHIFT,
            "method": "renoise",
        },
    )
    _require(manifest, "target_modules", list(PIKA_TARGET_MODULES))
    digest = manifest.get("sha256", "")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Pika manifest must include a lowercase hex SHA-256 digest")
    weights = root / "adapter.safetensors"
    if not path.is_dir() and path != weights:
        raise ValueError("A Pika manifest applies only to its sibling adapter.safetensors")
    if not weights.is_file():
        raise ValueError(f"Pika adapter weights not found: {weights}")
    return PikaSpec(
        task=task,
        sha256=digest,
        release_id=manifest["release_id"],
        numerics_policy=manifest["numerics_policy"],
        weights_path=str(weights),
    )


def verify_pika_weights(spec: PikaSpec) -> None:
    digest = hashlib.sha256()
    with open(spec.weights_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != spec.sha256:
        raise ValueError(f"Pika adapter checksum mismatch: {spec.weights_path}")


class PikaNoise(RenoiseNoiseSource):
    """Independent initial, re-noise and augmentation streams for one request.

    Every draw happens in Pika's external layout (video ``[1, T, 24, H, W]``,
    stereo audio ``[1, T, 64]``) and is packed afterwards; drawing directly in
    row order would change the samples for the same seed.

    Sequence parallel: the denoise state (target rows, x0, re-noised rows) is
    replicated on every SP rank, and the DiT shards the packed sequence only
    inside its forward. Each rank therefore builds this object with the same
    seed and the *full* latent geometry and draws the full tensors itself; a
    CUDA Philox stream with equal seed/offset produces identical values on GPUs
    of the same model (same SM count), so the ranks stay bit-identical without a
    collective, exactly as the CPU-generator base path does. Mixed GPU models in
    one SP group are not supported.
    """

    def __init__(
        self,
        *,
        seed: int,
        device: torch.device | str,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
    ) -> None:
        self.seed = int(seed)
        self.device = torch.device(device)
        self.video_shape = (1, int(latent_t), 24, int(latent_h), int(latent_w))
        self.audio_shape = (1, int(audio_t), 64)
        self.initial = torch.Generator(device=self.device).manual_seed(self.seed)
        self.renoise = torch.Generator(device=self.device).manual_seed(self.seed + PIKA_RENOISE_SEED_OFFSET)
        self.augmentation = torch.Generator(device=self.device).manual_seed(self.seed + PIKA_AUGMENTATION_SEED_OFFSET)
        # Re-noise generator state at the start of the most recent step, so a
        # retried step replays the same noise instead of consuming new draws.
        self._renoise_step: int | None = None
        self._renoise_state: torch.Tensor | None = None

    def video(self, generator: torch.Generator, shape: Sequence[int] | None = None) -> torch.Tensor:
        """Draw ``[1, T, 24, H, W]`` and pack it into ``[T*(H/2)*(W/2), 96]`` rows."""
        noise = torch.randn(
            tuple(shape or self.video_shape), generator=generator, device=self.device, dtype=torch.float32
        )
        return minimax_h3_patchify_video_latent(noise.permute(0, 2, 1, 3, 4), patch_size=(1, 2, 2))

    def audio(self, generator: torch.Generator) -> torch.Tensor:
        """Draw ``[1, T, 64]`` and pack it channel-major into ``[2T, 32]`` rows."""
        noise = torch.randn(self.audio_shape, generator=generator, device=self.device, dtype=torch.float32)
        steps = self.audio_shape[1]
        return noise.reshape(steps, 2, 32).permute(1, 0, 2).reshape(2 * steps, 32).contiguous()

    def initial_rows(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Initial target noise: one generator, video first, then audio (``evaluation.py:115-117``)."""
        video = self.video(self.initial)
        audio = self.audio(self.initial)
        return video, audio

    def visual_anchor(
        self,
        rows: torch.Tensor,
        shapes: Sequence[Sequence[int]],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Noised visual reference rows ``0.999 * lat + (1 - 0.999) * eps``.

        One augmentation draw per visual condition, in packed order, shape
        ``[1, Fr, 24, Hr, Wr]``; Python-scalar coefficients as in
        ``wrapper.py:80-94``. ``dtype`` is the numerics policy's conditioning
        dtype: bf16_conditioning_v1 rounds the clean latents first.
        """
        rows = rows.to(device=self.device, dtype=dtype).float()
        output = []
        offset = 0
        t = PIKA_CONDITION_NOISE_AUG
        for latent_t, latent_h, latent_w in shapes:
            noise = self.video(self.augmentation, (1, int(latent_t), 24, int(latent_h), int(latent_w)))
            count = int(noise.shape[0])
            output.append(t * rows[offset : offset + count] + (1.0 - t) * noise)
            offset += count
        if offset != int(rows.shape[0]):
            raise ValueError(
                f"Pika visual condition rows ({rows.shape[0]}) do not match their latent shapes ({offset})"
            )
        return torch.cat(output)

    def draw(self, step_index: int, like: Sequence[torch.Tensor | None]) -> tuple[torch.Tensor | None, ...]:
        """Fresh re-noise for one step: video before audio, from the ``seed + 1`` stream."""
        if len(like) != 2:
            raise ValueError("Pika re-noising expects (video, audio) streams")
        step_index = int(step_index)
        if self._renoise_step == step_index and self._renoise_state is not None:
            self.renoise.set_state(self._renoise_state)  # retry: replay this step's draws
        else:
            self._renoise_step = step_index
            self._renoise_state = self.renoise.get_state()
        out: list[torch.Tensor | None] = []
        for index, reference in enumerate(like):
            if reference is None:
                out.append(None)
                continue
            noise = self.video(self.renoise) if index == 0 else self.audio(self.renoise)
            if tuple(noise.shape) != tuple(reference.shape):
                raise ValueError(
                    f"Pika re-noise shape {tuple(noise.shape)} does not match the request rows {tuple(reference.shape)}"
                )
            out.append(noise.to(device=reference.device))
        return tuple(out)


PIKA_DUMP_DIR_ENV = "VLLM_OMNI_PIKA_DUMP_DIR"


class PikaDebugDump:
    """Opt-in parity dump: ``VLLM_OMNI_PIKA_DUMP_DIR=/path`` saves one request's
    conditioning, initial rows and per-step rows on DiT rank 0 (CPU ``.pt``
    files), for comparison with Pika golden tensors. Off unless the env var is
    set; never used on base requests.
    """

    def __init__(self, root: str | None, *, rank: int, seed: int) -> None:
        self.dir: Path | None = None
        if root and rank == 0:
            import time

            self.dir = Path(root) / f"req_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 10**6:06d}_s{seed}"
            self.dir.mkdir(parents=True, exist_ok=True)

    def __bool__(self) -> bool:
        return self.dir is not None

    def save(self, name: str, **values: Any) -> None:
        if self.dir is None:
            return
        out = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in values.items()}
        torch.save(out, self.dir / f"{name}.pt")


PIKA_INJECT_DIR_ENV = "VLLM_OMNI_PIKA_INJECT_GOLDEN"
# Opt-in parity tool: ``VLLM_OMNI_PIKA_DEBUG_BASE_DIT=1`` keeps the Pika server
# mode (sampler, noise streams, numerics) but does not pin the adapter, so
# requests run the base DiT. Used to separate DiT kernel differences from LoRA
# kernel differences against a Pika run with its adapter disabled. Never for
# serving: outputs are not Pika H3 Flex samples.
PIKA_DEBUG_BASE_DIT_ENV = "VLLM_OMNI_PIKA_DEBUG_BASE_DIT"


def load_pika_golden_conditioning(
    root: str,
    *,
    text_tags: torch.Tensor,
    visual_condition: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Opt-in parity tool: Pika golden ``03_cond.pt`` (``dump_pika_golden.py``) as engine conditioning.

    ``VLLM_OMNI_PIKA_INJECT_GOLDEN=<golden dir>`` replaces this request's text
    hidden states and clean visual reference latents with Pika's, so the DiT,
    LoRA and sampler can be compared step by step without the text-encoder and
    VAE-encoder differences. Every rank loads the same file. Shapes and text
    tags must match the request exactly.
    """
    cond = torch.load(Path(root) / "03_cond.pt", map_location="cpu", weights_only=False)
    embeds = cond["prompt_embeds"][0]
    tags = cond["text_token_tags"].reshape(-1).long()
    same_tags = torch.equal(tags, text_tags.reshape(-1).long().cpu())
    if tuple(embeds.shape[:1]) != tuple(text_tags.shape[:1]) or not same_tags:
        raise ValueError("Pika golden text presentation does not match this request")
    rows = None
    if visual_condition is not None:
        parts = [
            minimax_h3_patchify_video_latent(ref["video_latents"].float().permute(0, 2, 1, 3, 4), patch_size=(1, 2, 2))
            for ref in cond["references"]
            if ref["kind"] in ("image", "video")
        ]
        rows = torch.cat(parts)
        if rows.shape != visual_condition.shape:
            raise ValueError(f"Pika golden visual rows {tuple(rows.shape)} != request {tuple(visual_condition.shape)}")
    return embeds, rows


def describe_pika_sigmas() -> str:
    video, audio = pika_sigmas(PIKA_VIDEO_SHIFT), pika_sigmas(PIKA_AUDIO_SHIFT)
    return "video=" + ",".join(f"{s:.8f}" for s in video) + " audio=" + ",".join(f"{s:.8f}" for s in audio)


__all__ = [
    "PIKA_AUDIO_SHIFT",
    "PIKA_AUGMENTATION_SEED_OFFSET",
    "PIKA_CORRELATED_SEED_OFFSETS",
    "PIKA_DEBUG_BASE_DIT_ENV",
    "PIKA_DUMP_DIR_ENV",
    "PIKA_INJECT_DIR_ENV",
    "PIKA_BASE_REVISION",
    "PIKA_KNOTS",
    "PIKA_NOISE_VERSION",
    "PIKA_RENOISE_SEED_OFFSET",
    "PIKA_SERVED_TASKS",
    "PIKA_VIDEO_SHIFT",
    "PikaDebugDump",
    "PikaNoise",
    "PikaSpec",
    "describe_pika_sigmas",
    "load_pika_golden_conditioning",
    "pika_sigmas",
    "read_pika_spec",
    "verify_pika_weights",
]

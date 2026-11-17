# Pika H3 Flex (Ref2VA) on MiniMax H3

> Six-step re-noising distillation of MiniMax H3 Ref2VA, served as a dedicated
> vLLM-Omni server mode

## Summary

- Adapter: [`Pika-Labs/Pika-H3-Flex`](https://huggingface.co/Pika-Labs/Pika-H3-Flex) `ref2va/`
  (`adapter.safetensors` + `manifest.json`, `pika-flex-ref2va-v1`)
- Base: original-layout [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) `Ref2VA/`
  (the manifest pins 42ed227; 0919fb68 carries byte-identical weights)
- Task: `ref2va` only (FL2VA is not ported yet)
- Sampling contract (`pika-flex-noise-v1`, Pika-H3-Flex f961d7e):
  knots `[1000, 750, 500, 250, 100, 40, 0]/1000`, video shift 12, audio shift 3,
  6 transformer forwards, `x0 = x + sigma_grid * v`, fresh re-noise per step
  (`seed + 1`), initial noise `seed`, reference augmentation `seed + 7919`, all
  drawn on the GPU in Pika's external layouts. Same seed, same noise as Pika.

## Serve

Point `--lora-path` at the adapter directory. The manifest (format, rank/alpha
128, knots, shifts, numerics policy, base identity, SHA-256 of the weights) is
validated at startup; the adapter becomes the pipeline's fixed, pinned dynamic
BF16 LoRA at scale 1 (never merged). Without such a `--lora-path`, base H3
behavior is unchanged.

Validated on H200 (eager mode): one GPU and USP2 at 480p, USP4 at 768p, and
`--step-execution` on USP2 (bit-identical video to request mode). USP2 shown,
with text-encoder TP and VAE patch parallel matching the USP degree:

```bash
vllm serve /path/to/MiniMax-H3/Ref2VA --omni --trust-remote-code \
  --task-type ref2va --lora-path /path/to/Pika-H3-Flex/ref2va \
  --num-gpus 2 --usp 2 --ring 1 --cfg-parallel-size 1 --text-encoder-tp-size 2 \
  --vae-patch-parallel-size 2 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN --cache-backend none --enforce-eager \
  --additional-config '{"minimax_h3_vae_build_fp32": true}'
```

Other USP degrees use the same flags with `N` in place of 2; check the outputs
against a one-GPU run before relying on them. Compile mode (without
`--enforce-eager`) was checked on one GPU at 480p: it runs with the dynamic
adapter, matches eager output at kernel-noise level, and is not faster than
eager.

Expect `Pika H3 Flex server mode: ...` and `Pika H3 Flex LoRA bound 312/312
modules` in the log. Rejected at startup: `--lora-scale` other than 1,
`lora_application=merged`, a distilled or FastH3 base checkpoint, any
`--cache-backend` other than `none`, and VDN / Open-H3-Regenerate / FastH3
fusion artifacts.

Image references are VAE-encoded under Pika's numerics (cuDNN TF32 allowed)
instead of the base Hopper FP32 policy, which alone moves the reference latents
by ~1% relative L2 against Pika's (~4e-4 with Pika's policy).

## Request

A normal Ref2VA `/v1/videos` request. Keep the prompt byte-exact, including a
trailing newline (`curl -F "prompt=<file"` does, `$(cat file)` does not).
`num_inference_steps` may be omitted or 6, `flow_shift`/`audio_flow_shift`
omitted or 12/3. Rejected: other step counts or shifts, `sampler`, `quality=high` (Cache-DiT), request-level `lora`,
continuation / `continue_from`, latent refine, latent-mask editing, timeline
guides, control, `audio_mode=lock_source`, and `num_outputs_per_prompt` other
than 1. `latent_upscale` (post-denoise) is allowed. Under `--step-execution`,
`num_inference_steps=6` must be explicit; without it the step scheduler fails
the request before the pipeline sees it (a server error, as for any model).

Seeds: under `pika-flex-noise-v1` seed `S` draws its re-noise from the stream
that seed `S + 1` uses for its initial noise (and its augmentation from the
stream of `S + 7919`), so samples whose seeds differ by 1, 7918 or 7919 are
correlated. This is Pika's own convention. For batches, send one request per
seed and space the seeds apart, e.g. `base + k * 10007`; a multi-output request
(which would seed outputs `seed + k`) is rejected for that reason.

```bash
curl -sS http://127.0.0.1:8000/v1/videos/sync \
  -F "prompt=<prompt.txt" -F width=832 -F height=480 -F fps=24 -F seed=42 \
  -F 'extra_params={"task": "ref2va", "duration": 5.0}' \
  -F input_references=@ref_01.png -o out.mp4
```

Reference images keep vLLM-Omni's native size rule (original size rounded to
32). Pika resizes image references to a tiered short edge (1024 for 480p
canvases, 2048 for 768p and above, upscaling allowed); to reproduce Pika's
inputs, resize on the client with that rule and upload lossless PNGs.

## Numerics flags (`--additional-config`, default off)

- `minimax_h3_vae_build_fp32: true` - build video/audio VAEs under an FP32
  default dtype (the loader's ambient dtype is BF16, which rounds FP32 VAE
  weights and construction-time constants before they are cast back).
  Recommended for Pika serving: no runtime cost, audio closer to Pika.
- `minimax_h3_vae_decode_fp32: true` - decode video in FP32 without FP16
  autocast and without the FP16 fused decoder path (Pika's decode contract).
  It does not touch audio, which already decodes in FP32, and roughly doubles
  end-to-end time at 480p on one GPU; leave it off unless you are checking
  parity.

## Parity tooling (debug only)

- `VLLM_OMNI_PIKA_DUMP_DIR=<dir>`: DiT rank 0 saves conditioning, initial rows
  and per-step rows of each Pika request.
- `VLLM_OMNI_PIKA_INJECT_GOLDEN=<golden dir>`: replace text hidden states and
  clean reference latents with a Pika golden `03_cond.pt` to compare the DiT,
  LoRA and sampler alone.
- `VLLM_OMNI_PIKA_DEBUG_BASE_DIT=1`: keep the Pika sampler and noise but do not
  apply the adapter (base DiT), to compare against a Pika run with its adapter
  disabled. Outputs are not Pika samples.

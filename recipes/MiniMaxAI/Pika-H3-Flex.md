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
BF16 LoRA at scale 1 (never merged). Without such a `--lora-path` nothing
changes for base H3.

```bash
vllm serve /path/to/MiniMax-H3/Ref2VA --omni --trust-remote-code \
  --task-type ref2va --lora-path /path/to/Pika-H3-Flex/ref2va \
  --num-gpus 8 --usp 8 --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN --cache-backend none
```

Expect `Pika H3 Flex server mode: ...` and `Pika H3 Flex LoRA bound 312/312
modules` in the log. Do not pass `--lora-scale`, `lora_application=merged` or a
distilled base checkpoint; they are rejected.

## Request

A normal Ref2VA `/v1/videos` request. Keep the prompt byte-exact, including a
trailing newline (`curl -F "prompt=<file"` does, `$(cat file)` does not).
`num_inference_steps` may be omitted or 6, `flow_shift`/`audio_flow_shift`
omitted or 12/3. Rejected: other step counts or shifts, `sampler`, `quality=high` (Cache-DiT), request-level `lora`,
continuation / `continue_from`, latent refine, latent-mask editing, timeline
guides, control, `audio_mode=lock_source`. `latent_upscale` (post-denoise) is
allowed. Under `--step-execution`, `num_inference_steps=6` must be explicit.

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

## Optional numerics flags (`--additional-config`, default off)

- `minimax_h3_vae_build_fp32: true` - build video/audio VAEs under an FP32
  default dtype (the loader's ambient dtype is BF16, which rounds FP32 VAE
  weights and construction-time constants before they are cast back).
- `minimax_h3_vae_decode_fp32: true` - decode video in FP32 without FP16
  autocast and without the FP16 fused decoder path (Pika's decode contract;
  slower).

## Parity tooling (debug only)

- `VLLM_OMNI_PIKA_DUMP_DIR=<dir>`: DiT rank 0 saves conditioning, initial rows
  and per-step rows of each Pika request.
- `VLLM_OMNI_PIKA_INJECT_GOLDEN=<golden dir>`: replace text hidden states and
  clean reference latents with a Pika golden `03_cond.pt` to compare the DiT,
  LoRA and sampler alone.

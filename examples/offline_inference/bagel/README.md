# BAGEL-7B-MoT

## Setup
Please refer to the [stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/) to configure memory allocation appropriately for your hardware setup.

## Run examples

### Single Prompt
Get into the example folder
```bash
cd examples/offline_inference/bagel
```
Then run the command below.
```bash
bash run_single_prompt.sh
```

### Modality Control
BAGEL-7B-MoT supports multiple modality modes. You can control the mode using the `--modality` argument:

#### Text to Image (text2img)
Generate images from text prompts:
```bash
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality text2img \
                  --prompts "A cute cat"
```

#### Image to Image (img2img)
Transform images based on text prompts:
```bash
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality img2img \
                  --image-path /path/to/image.jpg \
                  --prompts "Make it more colorful"
```

#### Image to Text (img2text)
Generate text descriptions from images:
```bash
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality img2text \
                  --image-path /path/to/image.jpg \
                  --prompts "Describe this image in detail"
```

#### Text to Text (text2text)
Pure text generation:
```bash
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality text2text \
                  --prompts "What is the capital of France?"
```

### Using Prompt Files
You can load prompts from a text file (one prompt per line):
```bash
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality text2img \
                  --txt-prompts /path/to/prompts.txt
```

### Inference Steps
Control the number of inference steps for image generation:
```bash
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality text2img \
                  --steps 50 \
                  --prompts "A beautiful sunset"
```

## Supported Modalities
- `text2img`: Generate images from text prompts
- `img2img`: Transform images using text guidance (Stage 1 only)
- `img2text`: Generate text descriptions from images
- `text2text`: Pure text generation

## FAQ

If you encounter error about backend of librosa, try to install ffmpeg with the command below.
```bash
sudo apt update
sudo apt install ffmpeg
```

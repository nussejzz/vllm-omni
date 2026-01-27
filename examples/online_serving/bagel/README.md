# BAGEL-7B-MoT

## 🛠️ Installation

Please refer to [README.md](../../../README.md)

## Run examples (BAGEL-7B-MoT)

### Launch the Server

```bash
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni --port 8091
```

Or use the convenience script:
```bash
bash run_server.sh
```

If you have a custom stage configs file, launch the server with the command below:
```bash
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni --port 8091 --stage-configs-path /path/to/stage_configs_file
```

### Send Multi-modal Request

Get into the example folder:
```bash
cd examples/online_serving/bagel
```

#### Send request via Python

```bash
python openai_chat_client.py --prompt "A cute cat" --modality text2img
```

The Python client supports the following command-line arguments:

- `--prompt` (or `-p`): Text prompt for generation (default: `A cute cat`)
- `--output` (or `-o`): Output file path for image results (default: `bagel_output.png`)
- `--server` (or `-s`): Server URL (default: `http://localhost:8091`)
- `--image-url` (or `-i`): Input image URL or local file path (for img2img/img2text modes)
- `--modality` (or `-m`): Task modality (default: `text2img`). Options: `text2img`, `img2img`, `img2text`, `text2text`
- `--height`: Image height in pixels (default: 512)
- `--width`: Image width in pixels (default: 512)
- `--steps`: Number of inference steps (default: 25)
- `--seed`: Random seed (default: 42)
- `--negative`: Negative prompt for image generation

## Modality Control

BAGEL-7B-MoT supports multiple modality modes for different use cases.

### Supported Modalities

| Modality | Input | Output | Description |
|----------|-------|--------|-------------|
| `text2img` | Text | Image | Generate images from text prompts |
| `img2img` | Image + Text | Image | Transform images using text guidance |
| `img2text` | Image + Text | Text | Generate text descriptions from images |
| `text2text` | Text | Text | Pure text generation |

### Text to Image (text2img)

Generate images from text prompts:

#### Using Python client

```bash
python openai_chat_client.py \
    --prompt "A beautiful sunset over mountains" \
    --modality text2img \
    --output sunset.png \
    --steps 50
```

#### Using curl

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": [{"type": "text", "text": "<|im_start|>A beautiful sunset over mountains<|im_end|>"}]}],
    "modalities": ["image"],
    "num_inference_steps": 50
  }'
```

### Image to Image (img2img)

Transform images based on text prompts:

#### Using Python client

```bash
python openai_chat_client.py \
    --prompt "Make it more colorful and vibrant" \
    --modality img2img \
    --image-url /path/to/input.jpg \
    --output transformed.png
```

#### Using curl

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "<|im_start|>Make it more colorful<|im_end|>"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]
    }],
    "modalities": ["image"]
  }'
```

### Image to Text (img2text)

Generate text descriptions from images:

#### Using Python client

```bash
python openai_chat_client.py \
    --prompt "Describe this image in detail" \
    --modality img2text \
    --image-url /path/to/image.jpg
```

#### Using curl

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "<|im_start|>user\n<|image_pad|>\nDescribe this image<|im_end|>\n<|im_start|>assistant\n"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]
    }],
    "modalities": ["text"]
  }'
```

### Text to Text (text2text)

Pure text generation:

#### Using Python client

```bash
python openai_chat_client.py \
    --prompt "What is the capital of France?" \
    --modality text2text
```

#### Using curl

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": [{"type": "text", "text": "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"}]}],
    "modalities": ["text"]
  }'
```

## Using OpenAI Python SDK

### Text to Image

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8091/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="ByteDance-Seed/BAGEL-7B-MoT",
    messages=[{
        "role": "user",
        "content": [{"type": "text", "text": "<|im_start|>A cute cat<|im_end|>"}]
    }],
    extra_body={"modalities": ["image"]}
)

# Extract and save image from response
content = response.choices[0].message.content
if isinstance(content, list) and "image_url" in content[0]:
    img_data = content[0]["image_url"]["url"].split(",")[1]
    with open("output.png", "wb") as f:
        f.write(base64.b64decode(img_data))
```

### Image to Text

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="ByteDance-Seed/BAGEL-7B-MoT",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "<|im_start|>user\n<|image_pad|>\nDescribe this image<|im_end|>\n<|im_start|>assistant\n"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }],
    extra_body={"modalities": ["text"]}
)

print(response.choices[0].message.content)
```

## Generation Parameters

You can customize image generation with these parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `height` | Image height in pixels | 512 |
| `width` | Image width in pixels | 512 |
| `num_inference_steps` | Number of diffusion steps | 25 |
| `seed` | Random seed for reproducibility | None |
| `negative_prompt` | Negative prompt for image generation | None |

Example with custom parameters:

```bash
python openai_chat_client.py \
    --prompt "A futuristic city" \
    --modality text2img \
    --height 768 \
    --width 768 \
    --steps 50 \
    --seed 42 \
    --negative "blurry, low quality"
```

## FAQ

If you encounter an error about the backend of librosa, try to install ffmpeg with the command below:
```bash
sudo apt update
sudo apt install ffmpeg
```

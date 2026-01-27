import base64

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="ByteDance-Seed/BAGEL-7B-MoT",
    messages=[{"role": "user", "content": [{"type": "text", "text": "<|im_start|>A cute cat<|im_end|>"}]}],
    extra_body={"modalities": ["image"]},
)

# Extract and save image from response
content = response.choices[0].message.content
if isinstance(content, list) and "image_url" in content[0]:
    img_data = content[0]["image_url"]["url"].split(",")[1]
    with open("output.png", "wb") as f:
        f.write(base64.b64decode(img_data))

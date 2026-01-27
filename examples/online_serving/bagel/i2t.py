import base64

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="EMPTY")

# base64
with open("/workspace/vllm-omni/examples/online_serving/bagel/cat.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="ByteDance-Seed/BAGEL-7B-MoT",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<|im_start|>user\n<|image_pad|>\nDescribe this image<|im_end|>\n<|im_start|>assistant\n",
                },
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            ],
        }
    ],
    extra_body={"modalities": ["text"]},
)

print(response.choices[0].message.content)

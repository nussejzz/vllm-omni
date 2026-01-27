prompt="<|im_start|>user\nA cute cat<|im_end|>\n<|im_start|>assistant\n"

python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality text2img \
                  --prompt_type text \
                  --init-sleep-seconds 0 \
                  --prompts "${prompt}"

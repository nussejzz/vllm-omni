prompt="Please describe this image."
image_path="woman.png"

python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality img2text \
                  --image-path ${image_path} \
                  --prompts "${prompt}"

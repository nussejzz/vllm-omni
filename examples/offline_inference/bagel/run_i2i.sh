prompt="Let the woman wear a blue dress"
image_path="woman.png"

python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality img2img \
                  --image-path ${image_path} \
                  --prompts "${prompt}"
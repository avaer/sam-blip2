import wget
import os
sam_checkpoint = "sam_vit_h_4b8939.pth"

# if sam_checkpoint not found in the current directory, download it from wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
if not os.path.exists(sam_checkpoint):

    wget.download("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")

device = "cuda"
model_type = "default"

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# mask_generator = SamAutomaticMaskGenerator(sam)

def extract_masks(image):
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator.generate(image)

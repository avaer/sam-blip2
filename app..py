import io

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from starlette.responses import StreamingResponse

from utils import filter_segmentation, remove_overlaps
from sam import extract_masks
from blip2 import get_segment_captions
import matplotlib.pyplot as plt
import matplotlib.patches as patches


app = FastAPI()

@app.post("/get_labeled_bbox")
def get_labeled_bbox(img_file: UploadFile = File(...)):

    pil_image = Image.open(img_file.file).convert("RGB")
    image = np.array(pil_image)
    print("Extracting captioned Bounding boxes for image")

    # extract masks
    print("Extracting masks")
    masks = extract_masks(image)
    print("Extracted", len(masks), "masks")

    # filter masks
    image_area = image.shape[0] * image.shape[1]
    lower_area = image_area * (0.05 ** 2)
    upper_area = image_area * (0.8 ** 2)
    masks = filter_segmentation(masks, lower_area, upper_area)
    masks = remove_overlaps(masks, 0.01)
    print("Filtered masks to", len(masks))

    # get captions
    print("Getting captions")
    captions = get_segment_captions(image, masks)
    print("Got", len(captions), "captions")

    # show results
    return {"captions": captions}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
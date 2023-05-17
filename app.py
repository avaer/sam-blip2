import io

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from starlette.responses import StreamingResponse

from utils import filter_segmentation, remove_overlaps
from sam import extract_masks
from blip2 import get_segment_captions
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from starlette.responses import JSONResponse
import json
from datetime import datetime
import os

from segment_anything.utils.amg import batched_mask_to_box

app = FastAPI()

@app.post("/get_labeled_bbox")
def get_labeled_bbox(img_file: UploadFile = File(...)):

    pil_image = Image.open(img_file.file).convert("RGB")
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    print("Extracting captioned Bounding boxes for image")

    # extract masks
    print("Extracting masks")
    time = datetime.now()
    masks = extract_masks(image)
    endTime = datetime.now()
    timeDiff = endTime - time
    print("Extracted", len(masks), "masks in", timeDiff.total_seconds(), "seconds")

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
    fig, ax = plt.subplots(figsize=(20, 20))

    for mask in masks[:int(len(masks))]:
        x, y, w, h = mask['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        x = x - 10 if x - 10 > 0 else 0
        y = y - 10 if y - 10 > 0 else 0
        w = w + 10 if x + w + 10 < image.shape[1] else w
        h = h + 10 if y + h + 10 < image.shape[0] else h

        # draw bounding box and caption using matplotlib
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

        # put image caption in the center left of the bounding box
        ax.text(x, y + h/2, mask['caption'], fontsize=12, color='blue', horizontalalignment='left', verticalalignment='center')

    # show image with all bounding boxes
    ax.imshow(image)
    ax.axis('off')

    # Create a new image with the annotations added
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    new_image = np.asarray(buf)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGR)

    # turn new image to PIL image
    new_image = Image.fromarray(new_image)
    image_buffer = io.BytesIO()
    new_image.save(image_buffer, format="JPEG")
    image_buffer.seek(0)
    return StreamingResponse(image_buffer, media_type="image/jpeg")


@app.post("/get_boxes")
def get_boxes(img_file: UploadFile = File(...)):
    pil_image = Image.open(img_file.file).convert("L")
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Resize the image
    height, width = image.shape[:2]
    # Define the new width and height
    new_width = 512
    new_height = int(new_width * (height / width))

    # Resize the image
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    print("Extracting captioned Bounding boxes for image")

    # extract masks
    print("Extracting masks")
    time = datetime.now()
    masks = extract_masks(image)
    endTime = datetime.now()
    timeDiff = endTime - time
    print("Extracted", len(masks), "masks in", timeDiff.total_seconds(), "seconds")

    # filter masks
    image_area = image.shape[0] * image.shape[1]
    lower_area = image_area * (0.05 ** 2)
    upper_area = image_area * (0.8 ** 2)
    masks = filter_segmentation(masks, lower_area, upper_area)
    masks = remove_overlaps(masks, 0.01)
    print("Filtered masks to", len(masks))

    # convert masks to json
    json_masks = []
    for mask in masks:
        json_masks.append(mask['bbox'])
    
    # map the bounding boxes back to the original image size
    for json_mask in json_masks:
        json_mask[0] = int(json_mask[0] * (width / new_width))
        json_mask[1] = int(json_mask[1] * (height / new_height))
        json_mask[2] = int(json_mask[2] * (width / new_width))
        json_mask[3] = int(json_mask[3] * (height / new_height))

    # respond with json. make sureto set the content type to application/json
    response = JSONResponse(content=json_masks)
    response.headers["content-type"] = "application/json"
    return response


# def get_point_mask(image, x, y):
#     predictor = SamPredictor(sam)

#     predictor.set_image(image)

#     input_point = np.array([[x, y]])
#     input_label = np.array([1])

#     masks, scores, logits = predictor.predict(
#         point_coords=input_point,
#         point_labels=input_label,
#         multimask_output=True,
#     )
#     # masks.shape  # (number_of_masks) x H x W
#     number_of_masks = masks.shape[0]

#     return [masks, scores, logits]

@app.post("/get_point_mask")
def get_point_mask(x: str = Form(...), y: str = Form(...), img_file: UploadFile = Form(...)):
    pil_image = Image.open(img_file.file).convert("RGB")
    image = np.array(pil_image)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    print("image shape:", image.shape, "x:", x, "y:", y)

    width = image.shape[1]
    height = image.shape[0]

    # get x and y parameters (int). they must exist.
    # x = int(request.form.get("x"))
    # y = int(request.form.get("y"))
    # if x is None or y is None:
    #     return JSONResponse(content={"error": "x and y parameters must exist and be integers"}, status_code=400)

    # extract masks
    print("getting point masks")
    time = datetime.now()
    masks, scores, logits = extract_point_mask(image, x, y)
    endTime = datetime.now()
    timeDiff = endTime - time
    num_masks = masks.shape[0]
    print("got point masks (", timeDiff, ") in", timeDiff.total_seconds(), "seconds")

    # top mask based on score
    top_mask = None
    top_score = 0
    for i in range(num_masks):
        if scores[i] > top_score:
            top_mask = masks[i]
            top_score = scores[i]

    # get the top mask bbox
    top_mask_bbox = batched_mask_to_box(top_mask)
    top_mask_bbox_json_string = json.dumps(top_mask_bbox)

    # convert the top mask to a uint8 array
    top_mask = top_mask.astype(np.uint8)
    # get the bytes of the uint8array, for sending to the client
    top_mask_bytes = top_mask.tobytes()

    response = Response(content=top_mask_bytes)
    response.headers["content-type"] = "application/octet-stream"
    response.headers["X-Dims"] = json.dumps([width, height])
    response.headers["X-Box"] = top_mask_bbox_json_string
    return response

# get the port from the environment
port = int(os.environ.get("PORT", 8111))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=port)
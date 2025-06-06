import io

import cv2
import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, Response, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from utils import filter_segmentation, filter_confidence, filter_area_confidence, remove_overlaps
from sam import extract_masks, extract_point_mask
from blip2 import get_segment_captions
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from starlette.responses import JSONResponse
import json
from datetime import datetime
import os

from segment_anything.utils.amg import batched_mask_to_box
from PIL import Image, ImageOps

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # ← or list specific origins
    allow_credentials=True,
    allow_methods=["*"],        # ← GET, POST, PUT, DELETE, OPTIONS, …
    allow_headers=["*"],        # ← Authorization, Content-Type, …
    expose_headers=["X-Dims", "X-Bbox", "X-Num-Masks"],
)

@app.post("/get_labeled_bbox")
def get_labeled_bbox(img_file: UploadFile = File(...)):

    pil_image = Image.open(img_file.file).convert("RGB")
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # print("Extracting captioned Bounding boxes for image")

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
    # masks = remove_overlaps(masks, 0.01)
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
    response.headers["x-dims"] = json.dumps([width, height])
    return response


# @app.post("/get_boxes_raw")
# def get_boxes(img_file: UploadFile = File(...)):
#     pil_image = Image.open(img_file.file).convert("L")
#     image = np.array(pil_image)
#     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

#     # Resize the image
#     height, width = image.shape[:2]
#     # Define the new width and height
#     new_width = 512
#     new_height = int(new_width * (height / width))

#     # Resize the image
#     image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
#     print("Extracting captioned Bounding boxes for image")

#     # extract masks
#     print("Extracting masks")
#     time = datetime.now()
#     masks = extract_masks(image)
#     endTime = datetime.now()
#     timeDiff = endTime - time
#     print("Extracted", len(masks), "masks in", timeDiff.total_seconds(), "seconds")

#     # filter masks
#     image_area = image.shape[0] * image.shape[1]
#     lower_area = image_area * (0.05 ** 2)
#     upper_area = image_area * (0.8 ** 2)
#     masks = filter_segmentation(masks, lower_area, upper_area)
#     masks = remove_overlaps(masks, 0.01)
#     print("Filtered masks to", len(masks))

#     # convert masks to json
#     json_masks = []
#     for mask in masks:
#         json_masks.append(mask['bbox'])
    
#     # map the bounding boxes back to the original image size
#     for json_mask in json_masks:
#         json_mask[0] = int(json_mask[0] * (width / new_width))
#         json_mask[1] = int(json_mask[1] * (height / new_height))
#         json_mask[2] = int(json_mask[2] * (width / new_width))
#         json_mask[3] = int(json_mask[3] * (height / new_height))

#     # respond with json. make sureto set the content type to application/json
#     response = JSONResponse(content=json_masks)
#     response.headers["content-type"] = "application/json"
#     return response



@app.post("/get_point_mask")
def get_point_mask(points: str = Form(None), labels: str = Form(None), bbox: str = Form(None), img_file: UploadFile = Form(...)):
    # pil_image = Image.open(img_file.file)
    # # remove exif transpose
    # pil_image = ImageOps.exif_transpose(pil_image)
    # pil_image = pil_image.convert("RGB")
    # image = np.array(pil_image)
    # # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # read image with cv2, as RGB
    image = cv2.imdecode(np.fromstring(img_file.file.read(), np.uint8), cv2.IMREAD_COLOR)
    # ensure image is RGB. in case it had an alpha channel, remove it.
    if image.shape[2] == 4:
        image = image[:, :, :3]
    # convert to numpy array
    image = np.array(image)

    width = image.shape[1]
    height = image.shape[0]

    # parse arguments
    if points is not None:
        points = json.loads(points)
    else:
        points = []
    if labels is not None:
        labels = json.loads(labels)
    else:
        labels = [1] * len(points)
    # parse bbox json
    if bbox is not None:
        bbox = json.loads(bbox)

    # print("got point and bbox", point, bbox)

    # get x and y parameters (int). they must exist.
    # x = int(request.form.get("x"))
    # y = int(request.form.get("y"))
    # if x is None or y is None:
    #     return JSONResponse(content={"error": "x and y parameters must exist and be integers"}, status_code=400)

    # extract masks
    print("getting point masks")
    time = datetime.now()
    masks, scores, logits = extract_point_mask(image, points, labels, bbox)
    endTime = datetime.now()
    timeDiff = endTime - time
    num_masks = masks.shape[0]
    print("got point masks n = ", str(num_masks), " in", timeDiff.total_seconds(), "seconds")

    # top mask based on score
    top_mask = None
    top_score = 0
    for i in range(num_masks):
        if scores[i] > top_score:
            top_mask = masks[i]
            top_score = scores[i]

    # get the top mask bbox
    # convert to torch.Tensor
    top_mask_tensor = torch.from_numpy(top_mask)
    top_mask_bbox = batched_mask_to_box(top_mask_tensor)
    # convert back to numpy array
    top_mask_bbox = top_mask_bbox.numpy().tolist()
    top_mask_bbox_json_string = json.dumps(top_mask_bbox)

    # convert the top mask to a uint8 array
    top_mask = top_mask.astype(np.uint8)
    # get the bytes of the uint8array, for sending to the client
    top_mask_bytes = top_mask.tobytes()

    # fastapi response
    response = Response(content=top_mask_bytes)
    # set headers
    response.headers["content-type"] = "application/octet-stream"
    response.headers["X-Dims"] = json.dumps([width, height])
    response.headers["X-Bbox"] = top_mask_bbox_json_string
    return response

@app.post("/get_all_masks")
def get_all_masks(img_file: UploadFile = File(...), max_masks: int = 16):
    # Read and convert image
    pil_image = Image.open(img_file.file).convert("RGB")
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get image dimensions
    height, width = image.shape[:2]

    # Extract masks
    print("Extracting masks")
    time = datetime.now()
    masks = extract_masks(image)
    endTime = datetime.now()
    timeDiff = endTime - time
    print("Extracted", len(masks), "masks in", timeDiff.total_seconds(), "seconds")

    # Filter masks
    image_area = image.shape[0] * image.shape[1]
    lower_area = image_area * (0.01)
    upper_area = image_area * (0.5)
    confidence_threshold = 0.25
    masks = filter_segmentation(masks, lower_area, upper_area, confidence_threshold)
    # masks = remove_overlaps(masks, 0.5)
    print("Filtered masks to", len(masks))
    
    # Limit masks to max_masks parameter
    if len(masks) > max_masks:
        masks = masks[:max_masks]
        print(f"Clamped masks to {max_masks}")

    # Create a flat array of all masks concatenated together
    # Each mask is flattened and appended to the array
    combined_mask = np.concatenate([mask['segmentation'].flatten() for mask in masks])
    
    # Convert to int8 and ensure contiguous memory layout
    combined_mask = np.ascontiguousarray(combined_mask.astype(np.uint8))
    combined_mask_bytes = combined_mask.tobytes()

    # Create response with binary data
    response = Response(content=combined_mask_bytes)
    response.headers["content-type"] = "application/octet-stream"
    response.headers["x-dims"] = json.dumps([width, height])
    response.headers["x-num-masks"] = str(len(masks))
    return response

# get the port from the environment
port = int(os.environ.get("PORT", 8111))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=port)

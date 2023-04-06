import cv2
from fastapi import FastAPI, File, UploadFile
from matplotlib import pyplot as plt, patches
from starlette.responses import StreamingResponse
from PIL import Image
import io
import numpy as np

from non_sams.utils import load_object_detector, detect_objects, display_boxes, filter_boxes, get_segment_captions, display_captions


app = FastAPI()

object_detector = load_object_detector()

@app.post("/get_labeled_image")
def get_labeled_image(img_file: UploadFile = File(...)):
    pil_image = Image.open(img_file.file).convert("RGB")
    image = np.array(pil_image)
    boxes, labels = detect_objects(image, object_detector, size=(-1, -1))
    filtered_boxes = filter_boxes(boxes, 0.3)
    captioned_boxes = get_segment_captions(image, filtered_boxes)

    # create results plot and return it
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(image)
    # Add bounding boxes and captions to the image
    for mask in captioned_boxes:
        # Get box coordinates
        x1, y1, x2, y2 = mask["bbox"]
        # Create rectangle patch
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='b', facecolor='none')
        # Add patch to the plot
        ax.add_patch(rect)
        # Add caption to the plot
        ax.text(x1, y1 + 20, mask["caption"], fontsize=9, color='b')
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


@app.post("/get_labeled_boxes")
def get_labeled_boxes(img_file: UploadFile = File(...)):
    pil_image = Image.open(img_file.file).convert("RGB")
    image = np.array(pil_image)
    boxes, labels = detect_objects(image, object_detector, size=(-1, -1))
    filtered_boxes = filter_boxes(boxes, 0.3)
    captioned_boxes = get_segment_captions(image, filtered_boxes)
    return captioned_boxes

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
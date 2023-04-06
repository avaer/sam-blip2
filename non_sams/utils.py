import torch
import torchvision
import numpy as np
import cv2
import requests

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from blip2_utils.inference import get_blip2_caption

def display_boxes(image, boxes):
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(image)
    # Add bounding boxes to the image
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box
        # Create rectangle patch
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        # Add patch to the plot
        ax.add_patch(rect)
    # Show plot
    plt.show()

def display_captions(image, masks):
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(image)
    # Add bounding boxes and captions to the image
    for mask in masks:
        # Get box coordinates
        x1, y1, x2, y2 = mask['bbox']
        # Create rectangle patch
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        # Add patch to the plot
        ax.add_patch(rect)
        # Add caption to the plot
        ax.text(x1, y1 + 20, mask['caption'], fontsize=9, color='r')
    # Show plot
    plt.show()


# Load pre-trained object detection model
def load_object_detector():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).cuda()
    model.eval()
    return model

# Use object detector to detect objects in image and get bounding boxes
def detect_objects(image, object_detector, size=(448, 448)):
    if size[0] == -1:
        size = (image.shape[0], size[1])
    if size[1] == -1:
        size = (size[0], image.shape[1])
    # Convert image to tensor and normalize
    og_h, og_w, _ = image.shape
    image = torchvision.transforms.functional.to_tensor(image)
    print(image.shape)
    image = torchvision.transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    image = torchvision.transforms.functional.resize(image, size)
    # Add batch dimension
    image = image.unsqueeze(0)
    # Use object detector to get bounding boxes
    with torch.no_grad():
        predictions = object_detector(image.cuda())
    # Extract bounding boxes and class labels
    boxes = predictions[0]['boxes'].cpu().numpy()
    # rescale back to original image size
    if size[0] != og_h or size[1] != og_w:
        boxes = boxes * np.array([og_w/size[0], og_h/size[1], og_w/size[0], og_h/size[1]])
    labels = predictions[0]['labels'].cpu().numpy()
    return boxes, labels

# Filter bounding boxes that removes the bigger bounding box if a smaller one already fills more than 70% of its bounding area
def filter_boxes(boxes, area_threshold=0.7):
    """
    Removes overlapping boxes and keeps only important boxes.
    If a bigger box overlaps with a smaller box by more than 70%, the bigger box is removed.
    """
    filtered_boxes = []
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    sorted_idxs = sorted(range(len(boxes)), key=lambda x: -areas[x])
    removed_idxs = []

    for i, idx in enumerate(sorted_idxs):
        if idx in removed_idxs:
            continue
        box_i = boxes[idx]
        filtered_boxes.append(box_i)
        for j in range(i + 1, len(sorted_idxs)):
            idx_j = sorted_idxs[j]
            if idx_j in removed_idxs:
                continue
            box_j = boxes[idx_j]
            iou = compute_iou(box_i, box_j)
            if iou >= area_threshold and areas[idx] > areas[idx_j]:
                removed_idxs.append(idx_j)
    return filtered_boxes

def compute_iou(box1, box2):
    """
    Computes the intersection over union (IoU) between two bounding boxes.

    Parameters:
        box1 (list): [x1, y1, x2, y2], where (x1, y1) and (x2, y2) are the top-left and bottom-right coordinates
                      of the first bounding box.
        box2 (list): [x1, y1, x2, y2], where (x1, y1) and (x2, y2) are the top-left and bottom-right coordinates
                      of the second bounding box.

    Returns:
        iou (float): The IoU value between the two bounding boxes.
    """
    # Calculate the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # If the boxes don't overlap, return 0
    if x2 < x1 or y2 < y1:
        return 0.0

    # Calculate intersection area
    intersection_area = (x2 - x1) * (y2 - y1)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def get_segment_captions(image, boxes):
    masks = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1 = x1 - 10 if x1 - 10 > 0 else 0
        y1 = y1 - 10 if y1 - 10 > 0 else 0
        x2 = x2 + 10 if x2 + 10 < image.shape[1] else x2
        y2 = y2 + 10 if y2 + 10 < image.shape[0] else y2
        cropped = image[y1:y2, x1:x2]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
        caption = get_blip2_caption(cropped)
        # success, image_bytes = cv2.imencode('.jpg', cropped)
        # image_bytes = image_bytes.tobytes()
        #
        # url = 'http://llama-server.webaverse.com:5447/caption'
        # headers = {'Content-Type': 'image/jpeg'}
        # response = requests.post(url, data=image_bytes, headers=headers)
        # caption = response.text
        mask = {'bbox': (x1, y1, x2, y2), 'caption': caption}
        masks.append(mask)

    return masks


import cv2
import numpy as np
from utils import filter_segmentation, remove_overlaps
from sam import extract_masks
from blip2 import get_segment_captions
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import click

# Load the image
@click.command()
@click.option('--image', default='./test.png', help='Path to the image', type=click.Path(exists=True))
def get_labeled_bbox(image):
    print("Extracting captioned Bounding boxes for image at", image)
    image = cv2.imread(image)

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
    fig, ax = plt.subplots(figsize=(20, 20))

    for mask in masks[:int(len(masks))]:
        x, y, w, h = mask['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        x = x - 10 if x - 10 > 0 else 0
        y = y - 10 if y - 10 > 0 else 0
        w = w + 10 if x + w + 10 < image.shape[1] else w
        h = h + 10 if y + h + 10 < image.shape[0] else h

        # draw bounding box and caption using matplotlib
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # put image caption in the center left of the bounding box
        ax.text(x, y + h/2, mask['caption'], fontsize=12, color='red', horizontalalignment='left', verticalalignment='center')

    # show image with all bounding boxes
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(image)
    ax.axis('off')
    plt.imsave('output.png', image)


if __name__ == '__main__':
    get_labeled_bbox()
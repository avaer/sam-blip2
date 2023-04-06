from non_sams.utils import load_object_detector, detect_objects, display_boxes, filter_boxes, get_segment_captions, display_captions

from PIL import Image
import numpy as np

object_detector = load_object_detector()
image = Image.open("../discord.png").convert("RGB")
image = np.array(image)
boxes, labels = detect_objects(image, object_detector, size=(-1, -1))
print(boxes, labels)
print("num boxes", len(boxes))

# display boxes
display_boxes(image, boxes)

filtered_boxes = filter_boxes(boxes, 0.3)
print(len(filtered_boxes))
display_boxes(image, filtered_boxes)

captioned_boxes = get_segment_captions(image, filtered_boxes)
print(captioned_boxes)
display_captions(image, captioned_boxes)
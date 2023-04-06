
def filter_segmentation(masks, lower_size_threshold, upper_size_threshold, confidence_threshold=0.9):
    """Filter segmentation masks based on size and confidence.

    Args:
        masks [dict[area, predicted_iou, bbox]]: Segmentation
        lower_size_threshold (float): Lower size threshold
        upper_size_threshold (float): Upper size threshold
        confidence_threshold (float): Confidence threshold
    """
    filtered_masks = []
    for mask in masks:
        area = mask['area']
        confidence = mask['predicted_iou']
        if lower_size_threshold < area < upper_size_threshold and confidence > confidence_threshold:
            filtered_masks.append(mask)
    return filtered_masks

def remove_overlaps(masks, intersection_threshold=0.5):
    """
    Remove overlapping bounding boxes.
    Args:
        masks (dict[area, confidence, bbox]): Segmentation
        intersection_threshold (float): Intersection threshold
    """
    # Iterate over a copy of the list of dictionaries
    for i, dict1 in enumerate(masks.copy()):
        # Iterate over all the other dictionaries in the copy of the list
        for j, dict2 in enumerate(masks[i+1:].copy(), start=i+1):
            bbox1 = dict1['bbox']
            bbox2 = dict2['bbox']

            # Calculate the area of overlap between the two bounding boxes
            x_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
            y_overlap = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
            overlap_area = x_overlap * y_overlap

            # Calculate the area of each bounding box
            bbox1_area = dict1['area']
            bbox2_area = dict2['area']

            # Calculate the percentage of overlap
            overlap_percentage = overlap_area / min(bbox1_area, bbox2_area)
            # Remove the bounding box with the higher score if they overlap by more than 30%
            if overlap_percentage > intersection_threshold:
                if dict1['area'] < dict2['area']:
                    try:
                        masks.remove(dict1)
                        break
                    except:
                        pass
                else:
                    try:
                        masks.remove(dict2)
                    except:
                        pass

    return masks


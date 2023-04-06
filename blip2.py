import cv2
import requests
# get the top 10% masks and crop the image to the bounding box of each mask and display it the bounding box values are in format xywh and need to be casted to ints and I want to expand them by 10 pixels in every direction if possible
def get_segment_captions(image, masks):
    for mask in masks[:int(len(masks))]:
        x,y,w,h = mask['bbox']
        x,y,w,h = int(x),int(y),int(w),int(h)
        x = x - 10 if x - 10 > 0 else 0
        y = y - 10 if y - 10 > 0 else 0
        w = w + 10 if x + w + 10 < image.shape[1] else w
        h = h + 10 if y + h + 10 < image.shape[0] else h
        cropped = image[y:y+h, x:x+w]
        success, image_bytes = cv2.imencode('.jpg', cropped)
        image_bytes = image_bytes.tobytes()

        url = 'http://llama-server.webaverse.com:5447/caption'
        headers = {'Content-Type': 'image/jpeg'}
        response = requests.post(url, data=image_bytes, headers=headers)
        mask['caption'] = response.text
    return masks


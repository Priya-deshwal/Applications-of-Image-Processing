import cv2
import numpy as np

# Usage
def solution(image_path):
    image= cv2.imread(image_path)
    ######################################################################
    ######################################################################
    '''
    The pixel values of output should be 0 and 255 and not 0 and 1
    '''
    #####  WRITE YOUR CODE BELOW THIS LINE ###############################

    image = cv2.imread(image_path)

    factor = 0.5
    image = image.astype(np.float32)
    image = image * (1.0 - factor)
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)

    thresholded_image = np.zeros_like(image)

    for i in range(3):  # Loop over the RGB channels (BGR in OpenCV)
       channel = image[:, :, i]  # Select the current channel
       _, binary_channel = cv2.threshold(channel, 65, 255, cv2.THRESH_BINARY)  # Adjust the threshold as needed
       thresholded_image[:, :, i] = binary_channel

    sky_detection = np.all(thresholded_image == 255, axis=2)

    image[sky_detection] = [0, 0, 0]

    h, s, v = cv2.split(image)
    _, thresh = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ig = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
    )
    ig = cv2.morphologyEx(
        ig, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8)
    )
    contours, _ = cv2.findContours(
        ig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    max_area = 0
    largest_contour = None
    for contour in contours:
      area = cv2.contourArea(contour)
      if area > max_area:
          max_area = area
          largest_contour = contour
    mask = np.zeros_like(ig)

    if largest_contour is not None:
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    lava = np.zeros_like(image)
    lava[mask == 255] = [255, 255, 255]

    
    ######################################################################  
    return lava

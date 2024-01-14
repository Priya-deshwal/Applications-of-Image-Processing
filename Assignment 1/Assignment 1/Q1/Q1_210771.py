import cv2
import numpy as np

# Usage
def solution(image_path):
    # image= cv2.imread(image_path)
    ######################################################################
    ######################################################################
    #####  WRITE YOUR CODE BELOW THIS LINE ###############################
    image1 = cv2.imread(image_path)
    padding_size = 10
    img_padded = cv2.copyMakeBorder(image1, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=0)

    # Convert the image to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to the padded grayscale image
    threshold_value = 10
    _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

# Apply padding to the grayscale image (e.g., with zero padding)
    padding_size = 10
    gray_padded = cv2.copyMakeBorder(thresholded, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=0)

# Apply thresholding to the padded grayscale image
#threshold_value = 100
#_, thresholded = cv2.threshold(gray_padded, threshold_value, 255, cv2.THRESH_BINARY)

# Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image
    image_with_contours = image.copy()
    #cv2.drawContours(image_with_contours, contours, -1, (0, 0, 255), 2)  # Draw in red color

# detect corners with the goodFeaturesToTrack function.
    corners = cv2.goodFeaturesToTrack(gray_padded, 4, 0.5, 100)
    corners = np.int0(corners)
  
# we iterate through each corner, 
# making a circle at each point that we think is a corner.
    for i in corners:
     x, y = i.ravel()
     cv2.circle(gray_padded, (x, y), 8, 255, -1)   


    corner_list = []  # Create an empty list to store corner coordinates

    for i in range(4):
     x=corners[i][0][0]
     y=corners[i][0][1]
     corner_list.append([x, y])

# Convert the list of corner coordinates to a NumPy array
    corner_array = np.array(corner_list)

    # Sort the points by y-coordinate.
    corner_array = corner_array[corner_array[:, 1].argsort()]

  # Sort the top two points by x-coordinate.
    top_two_points = corner_array[:2]
    top_two_points = top_two_points[top_two_points[:, 0].argsort()]

  # Sort the bottom two points by x-coordinate.
    bottom_two_points = corner_array[2:]
    bottom_two_points = bottom_two_points[bottom_two_points[:, 0].argsort()]

  # Combine the sorted points into a single array.
    sorted_points = np.concatenate((top_two_points, bottom_two_points))

    corner_array = sorted_points
    corner_array = np.float32(corner_array)

    width = 600
    height = 600
#corner = np.float32([[0, 0], [gray_padded.shape[1], 0], [0, gray_padded.shape[0]], [gray_padded.shape[1], gray_padded.shape[0]]])

#converted_points  =np.float32([[0,0], [width,0], [0,height], [width,height]])
#converted_points  =np.float32([[0,height], [width,height], [width,0], [0,0]])
    converted_points  =np.float32([[width,height], [0,height], [width,0], [0,0]])
    matrix = cv2.getPerspectiveTransform(corner_array, converted_points)
    img_output = cv2.warpPerspective(img_padded, matrix, (width, height))

#res_image = cv2.rotate(img_output, cv2.ROTATE_180)
#cv2_imshow(img_output)
    ######################################################################
    ######################################################################

    return img_output



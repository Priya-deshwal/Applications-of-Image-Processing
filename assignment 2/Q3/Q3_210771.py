import cv2
import numpy as np

def solution(audio_path):
    ############################
    ############################

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    img = cv2.imread(audio_path)

    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to identify non-white pixels
    _, mask = cv2.threshold(image, 210, 255, cv2.THRESH_BINARY)

    # Set non-white pixels to black
    result = cv2.bitwise_and(image, image, mask=mask)

    cntr_img = np.zeros_like(result)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(cntr_img, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    krnl = np.ones((5, 5), np.uint8)
    ig = cv2.morphologyEx(cntr_img, cv2.MORPH_OPEN, krnl)
    hgt, wdt = ig.shape

    click = ig
    # for i in range(wdt):
    #   if click[hgt - 1, i] == 0:
    #     left = i
    #     break
    # for j in range(wdt):
    #   if click[hgt - 1, wdt - j - 1] == 0:
    #     right = wdt - j - 1
    #     break
    # mid = (left + right) // 2
    left = next((i for i, value in enumerate(range(wdt)) if click[hgt - 1, i] == 0), None)
    right = next((wdt - j - 1 for j, value in enumerate(range(wdt)) if click[hgt - 1, wdt - j - 1] == 0), None)

    mid = (left + right) // 2 if left is not None and right is not None else None
       
    ratio = (wdt - mid) / mid
    # flag = 0
    if ratio > 1.08:
      # flag = 1
      class_name = "real"
      return class_name


    # if flag == 1:
    class_name = "fake"

    return class_name

import cv2
import os
import numpy as np

image_root = "/home/pimovergaag/PycharmProjects/bebop-gate-obstacle-detection/data/raw"
images = sorted(os.listdir(image_root))
for image in images:
    img = cv2.imread(os.path.join(image_root, image))
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 100, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    cv2.imshow("mask", mask)
    cv2.imshow("original", img)
    key = cv2.waitKey(0)
    if key == 27:
        break
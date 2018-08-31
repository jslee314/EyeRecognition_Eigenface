import numbers as np
import cv2

file = '01_image'
img = cv2.imread(file,cv2.IMREAD_COLOR)

cv2.imshow('show image',img)
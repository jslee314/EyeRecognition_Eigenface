import cv2
import imutils

org_img = cv2.imread('ljs.png')

# 1. find OpenCV functions by name
imutils.find_function("contour")
'''
1. contourArea
2. drawContours
3. findContours
4. isContourConvex

-> cv.contourArea
'''

# 2. Translation
translate_img = imutils.translate(org_img, 20, -75)
'''
cv.warpAffine 

x=25 pixels to the right,
y=75 pixels up
'''

# 3. Rotation
for angle in range(0, 360, 90):
    rotate_img = imutils.rotate(org_img, angle=angle)
    cv2.imshow("Angle: %d" % (angle), rotate_img)
'''
cv.gerRotationMatrix2D

'''
# 4. Resizing
for size in (400, 300, 200, 100):
    resize_img = imutils.resize(org_img, width=size, height=size/2)

# 5. Skeletonization
gray = cv2.cvtColor(org_img,cv2.COLOR_BGR2GRAY)
skeleton_image = imutils.skeletonize(gray, size=(2,3))
cv2.imshow("Skeleton", skeleton_image)

# 6. display with matplotlib
'''cv.imshow

'''


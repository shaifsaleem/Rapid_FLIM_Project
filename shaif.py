import cv2 as cv
import numpy as np

img = np.load('images\PE-LD_M01_RF000.npy')
iamge = (img*255).astype(np.uint8)
iamge = iamge[1,:,:]

cv.imshow('Image', iamge )

blur = cv.GaussianBlur(iamge, (7,7), cv.BORDER_DEFAULT)
cv.imshow('blurred', blur)

## Cannying
# Canny = cv.Canny(blur, 125, 175, cv.THRESH_BINARY)
# cv.imshow('Canny', Canny)
# contours, hierarchies = cv.findContours(Canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

## Thresholding
ret, thresh = cv.threshold(blur, 175, 255, cv.THRESH_BINARY)
cv.imshow('Thresholded image', thresh)
contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

dilated = cv.dilate(thresh, (3,3), iterations=5)
cv.imshow('Dilated', dilated)



print(f'{len(contours)} contour(s) found..!')
cv.waitKey(0)
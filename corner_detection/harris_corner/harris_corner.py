import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('board_chess.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray) # convert image into floating points
dst = cv2.cornerHarris(gray,2,3,0.04) # (where,blockSize,ksize,k)
'''
where --> imput image (gray scaled)
blockSize --> size of the neighbourhood considered for corner detection
ksize --> aperture parameter of sobel derivative
k --> quality
'''
dst = cv2.dilate(dst,None) # result is dilated for marking corners

img[dst > 0.01 * dst.max()] = [0,250,15]

cv2.imshow('dst',img)
cv2.imwrite('corners_detected.jpg',img)

cv2.waitKey()
cv2.destroyAllWindows()
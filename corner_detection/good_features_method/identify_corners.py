import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('corner_image.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray',gray)
cv2.imwrite('gray_image.jpg',gray)

corners = cv2.goodFeaturesToTrack(gray,100,0.01,10) # (where,maxCornersToDetect,quality,distanceBetweenCorners)
corners = np.int0(corners) # converts into integer values

for corner in corners:
	x,y = corner.ravel() # takes the inner list in a nested list
	cv2.circle(img,(x,y),3,255,-1)

cv2.imshow('corner',img)
cv2.imwrite('corners_detected.jpg',img)

cv2.waitKey()
cv2.destroyAllWindows()
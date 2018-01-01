import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('cycle.png')

# ROI = Region of Image
cut = img[155:300,10:120]
cut[:] = [96,60,62] # fill the region with a random color

# shifting the region
some_region = img[100:320,300:500]
img[0:220,0:200] = some_region

cv2.imwrite('roi_cycle.png',img)

cv2.imshow('cycle',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
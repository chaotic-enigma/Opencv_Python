
# video breaks down into frames, frames break down into impages

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('chemistry_cats.jpg',cv2.IMREAD_GRAYSCALE) # read the image into gray scale
# IMREAD_COLOR = 1
# IMREAD_UNCHANGED = -1

cv2.imshow('cats',img) # show the image
cv2.waitKey() # wait until 'any' key is pressed
cv2.destroyAllWindows()
cv2.imwrite('gray_cats.jpg',img) # save image

# show in matplotlib
plt.imshow(img,cmap='RdYlBu',interpolation='bicubic') # cmap='gray'
plt.xticks([])
plt.yticks([])
plt.savefig('ryb_cats.jpg')
plt.show()
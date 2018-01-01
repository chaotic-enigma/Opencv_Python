import cv2
import numpy as np
import matplotlib.pyplot as plt

img_rgb = cv2.imread('big_image.jpg')

template = cv2.imread('match.jpg')
cv2.imshow('template',template)

c,w,h = template.shape[::-1]

res = cv2.matchTemplate(img_rgb,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]): # all locations
	cv2.rectangle(img_rgb,pt,(pt[0] + w,pt[1] + h),(0,0,255),2)

cv2.imshow('img_rgb',img_rgb)
cv2.imwrite('multi_template_matched.jpg',img_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()
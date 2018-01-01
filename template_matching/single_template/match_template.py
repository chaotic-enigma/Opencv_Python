import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('small_chess.jpg')
template = cv2.imread('chessy_template.jpg')

c,w,h = template.shape[::-1]

res = cv2.matchTemplate(img,template,cv2.TM_SQDIFF) # TM_SQDIFF --> best method
min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res) # unpack and find the location

top_left = min_loc
bottom_right = (top_left[0] + w,top_left[1] + h)

cv2.rectangle(img,top_left,bottom_right,55,2) # draw a rectangle having color 255 and with thickness 2

cv2.imshow('chess',img)
cv2.imwrite('camel_matched.jpg',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
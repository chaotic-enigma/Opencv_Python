import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('anogamepy.png')
img2 = cv2.imread('game_image.png')
img3 = cv2.imread('python_logo.jpeg')

'''
add = img1 + img2
cv2.imwrite('lost_opacity.png',add)

# (155,211,79) + (50,170,200) = (205,381,279) ... translated to (205,255,255)
add = cv2.add(img1,img2)
cv2.imwrite('cv2_add_method.png',add)

weighted = cv2.addWeighted(img1,0.6,img2,0.4,0)
cv2.imwrite('added_weights.png',weighted)
'''

# put logo top left corner
rows,cols,channels = img3.shape
roi = img1[0:rows,0:cols] # region of img3 in img1 (top-left)

img3gray = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY) # convert img3 into gray
ret,mask = cv2.threshold(img3gray,230,255,cv2.THRESH_BINARY_INV) 
# 0 or 1, 230 -> threshold value (if the pixel value found > 230, translated to 255, else 0)
# _INV -> inverse colors

'''
cv2.imshow('mask',mask)
cv2.imwrite('binary_inv.jpeg',mask)
'''

# bitwise operators
mask_inv = cv2.bitwise_not(mask)
'''
cv2.imshow('inv',mask_inv)
cv2.imwrite('mask_inv.jpeg',mask_inv)
'''
img1_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
'''
cv2.imshow('img1_bg',img1_bg)
cv2.imwrite('img1_bg_merged.png',img1_bg)
'''
img3_fg = cv2.bitwise_and(img3,img3,mask=mask)
'''
cv2.imshow('img3_fg',img3_fg)
cv2.imwrite('img3_fg_merged.png',img3_fg)
'''

dst = cv2.add(img1_bg,img3_fg)
'''
cv2.imshow('dst',dst)
cv2.imwrite('dst_merged.png',dst)
'''
img1[0:rows,0:cols] = dst

cv2.imshow('final_img1',img1)
cv2.imwrite('successful_merge.png',img1)

cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(img1)
plt.xticks([])
plt.yticks([])
plt.savefig('plt_plot.png')
plt.show()
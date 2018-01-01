import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('rover_b_l.png',cv2.COLOR_BGR2GRAY)
cv2.imshow('rover',img)

# add blur (smoothening)
gaussian_blur = cv2.GaussianBlur(img,(5,5),0)
cv2.imshow('gaussian_blur',gaussian_blur)
cv2.imwrite('gaussian_blur.png',gaussian_blur)

# sobel - detect edges after smoothening
sobel_x = cv2.Sobel(gaussian_blur,cv2.CV_64F,1,0,ksize=-1)
cv2.imshow('sobel_x',sobel_x)
cv2.imwrite('sobel_x.png',sobel_x)

sobel_y = cv2.Sobel(gaussian_blur,cv2.CV_64F,0,1,ksize=-1)
cv2.imshow('sobel_y',sobel_y)
cv2.imwrite('sobel_y.png',sobel_y)

# add sobel_x and sobel_y
sobel_xy = cv2.addWeighted(sobel_x,0.6,sobel_y,0.4,0)
cv2.imshow('sobel_xy',sobel_xy)
cv2.imwrite('sobel_xy.png',sobel_xy)

# laplacian - detect edges after smoothening
laplacian = cv2.Laplacian(gaussian_blur,cv2.CV_64F)
cv2.imshow('laplacian',laplacian)
cv2.imwrite('laplacian.png',laplacian)

# canny - detect edges after smoothening
canny = cv2.Canny(gaussian_blur,100,100)
cv2.imshow('canny',canny)
cv2.imwrite('canny.png',canny)

cv2.waitKey(0)
cv2.destroyAllWindows()

titles = ['Original','Sobel_x','Sobel_y','Sobel_xy','Laplacian','Canny']
images = [img,sobel_x,sobel_y,sobel_xy,laplacian,canny]

for i in xrange(len(images)):
	plt.subplot(2,3,i+1)
	plt.title(titles[i])
	plt.imshow(images[i],cmap='gray',interpolation='bicubic')
	plt.xticks([])
	plt.yticks([])

plt.savefig('all_edges')
plt.show()
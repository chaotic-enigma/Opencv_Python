import cv2
import numpy as np
import matplotlib.pyplot as plt

# The idea of thresholding is some form of extreme simplification of an image
# threshold  -> everything in the image is 0 or 1 (most basic level)

img = cv2.imread('bookpage.jpg')
cv2.imshow('original',img)

#8 -> threshold value (if the pixel value found > 8, translated to 255, else 0)
retval,threshold = cv2.threshold(img,8,255,cv2.THRESH_BINARY)
cv2.imshow('threshold',threshold)
cv2.imwrite('normal_threshold.jpg',threshold)

gray_scaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_scaled',gray_scaled)
cv2.imwrite('original_gray_scaled.jpg',gray_scaled)
retval2,gray_threshold = cv2.threshold(gray_scaled,8,255,cv2.THRESH_BINARY)
cv2.imshow('gray_threshold',gray_threshold)
cv2.imwrite('gray_threshold.jpg',gray_threshold)

# OTSU Threshold
retval3,otsu = cv2.threshold(gray_scaled,8,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('otsu',otsu)
cv2.imwrite('otsu_threshold.jpg',otsu)

# Making the image proper by adavtive threshold methods

# Mean Adaptive Thresholding (decides how thresholding value is calculated)
# The image should be converted into gray scale first
mean_threshold = cv2.adaptiveThreshold(gray_scaled,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,105,1) # _C - constant subtracted from the mean or weighted mean calculated
cv2.imshow('mean_threshold',mean_threshold)
cv2.imwrite('mean_threshold.jpg',mean_threshold)

# Gaussian Adaptive Threshold (get adapted according to the pixel values)
# The image should be converted into gray scale first
gaussian_threshold = cv2.adaptiveThreshold(gray_scaled,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,105,2) # _C - constant subtracted from the mean or weighted mean calculated
cv2.imshow('gaussian_threshold',gaussian_threshold)
cv2.imwrite('gaussian_threshold.jpg',gaussian_threshold)

cv2.waitKey(0)
cv2.destroyAllWindows()

titles = ['Original Image','Global Thresholding (v = 8)','Adaptive Mean Thresholding','Adaptive Gaussian Thresholding']
images = [img,threshold,mean_threshold,gaussian_threshold]

for i in xrange(len(images)):
	plt.subplot(2,2,i+1)
	plt.imshow(images[i],cmap='gray',interpolation='bicubic')
	plt.title(titles[i])
	plt.xticks([])
	plt.yticks([])

plt.savefig('all_thresholds.jpg')
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('j.png',cv2.COLOR_BGR2GRAY)
cv2.imshow('j',img)

kernel = np.ones((5,5),np.uint8)
kernel_9 = np.ones((9,9),np.uint8)
#print(kernel,kernel_9)

erosion = cv2.erode(img,kernel,iterations=1)
cv2.imshow('erosion',erosion)
cv2.imwrite('j_erosion.jpg',erosion)

dilation = cv2.dilate(img,kernel,iterations=1)
cv2.imshow('dilation',dilation)
cv2.imwrite('j_dilation.jpg',dilation)

opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
cv2.imshow('opening',opening)
cv2.imwrite('j_opening.jpg',opening)

closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
cv2.imshow('closing',closing)
cv2.imwrite('j_closing.jpg',closing)

gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
cv2.imshow('gradient',gradient)
cv2.imwrite('j_gradient.jpg',gradient)

tophat = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel_9)
cv2.imshow('tophat',tophat)
cv2.imwrite('tophat.jpg',tophat)

blackhat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel_9)
cv2.imshow('blackhat',blackhat)
cv2.imwrite('blackhat.jpg',blackhat)

cv2.waitKey(0)
cv2.destroyAllWindows()

titles = ['Original','Erosion','Dilation','Opening','Closing','Gradient','Tophat','Blackhat']
images = [img,erosion,dilation,opening,closing,gradient,tophat,blackhat]

for i in xrange(len(images)):
	plt.subplot(2,4,i+1)
	plt.imshow(images[i])
	plt.title(titles[i])
	plt.xticks([])
	plt.yticks([])

plt.savefig('morphology_transformation.jpg')
plt.show()
'''
Morphological transformations to remove noise form the filters

erosion --> the gradual destruction or diminution of something (in this case, color of pixels in the image)
dilation --> opposite of erosion (basically, increasing or widening)
opening --> filters from the background (fasle positive)
closing --> filters from the foreground (false negative)

In case of noise removal, erosion is followed by dilation
'''
print(__doc__)

import cv2
import numpy as np

captured = cv2.VideoCapture(0)

while True:

	_,frame = captured.read()
	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

	lower_red = np.array([150,50,50])
	upper_red = np.array([180,255,255])

	mask = cv2.inRange(hsv,lower_red,upper_red)
	res = cv2.bitwise_and(frame,frame,mask=mask)

	kernel = np.ones((5,5),np.uint8)

	# kernel interaction with mask
	erosion = cv2.erode(mask,kernel,iterations=1)
	dilation = cv2.dilate(mask,kernel,iterations=1)

	opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
	closing = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)

	cv2.imshow('frame',frame)
	cv2.imshow('res',res)
	cv2.imshow('erosion',erosion)
	cv2.imshow('dilation',dilation)
	cv2.imshow('opening',opening)
	cv2.imshow('closing',closing)

	kill = cv2.waitKey(1) & 0xFF
	if kill == ord('q'):
		break

captured.release()
cv2.destroyAllWindows()

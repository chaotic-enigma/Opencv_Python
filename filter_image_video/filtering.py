import cv2
import numpy as np

captured = cv2.VideoCapture(0)

while True:

	_,frame = captured.read()
	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) # hsv - Hue Saturation Value

	lower_red = np.array([150,50,50])
	upper_red = np.array([180,255,255])

	mask = cv2.inRange(hsv,lower_red,upper_red)
	res = cv2.bitwise_and(frame,frame,mask=mask) # where there is something in the frame where the mask (0 or 1) is true
	# if mask in range -> 1 (white), else 0

	kernel = np.ones((15,15),np.float32)/255
	smoothed = cv2.filter2D(res,-1,kernel)# filtering the noise with a kernel

	gaussin_blur = cv2.GaussianBlur(res,(15,15),0)
	median_blur = cv2.medianBlur(res,15)
	bilateral_blur = cv2.bilateralFilter(res,9,75,75)

	cv2.imshow('frame',frame)
	#cv2.imshow('mask',mask)
	cv2.imshow('res',res)
	#cv2.imshow('smoothed',smoothed)
	cv2.imshow('blur',gaussin_blur)
	cv2.imshow('median_blur',median_blur)
	cv2.imshow('bilateral_blur',bilateral_blur)

	kill = cv2.waitKey(1) & 0xFF
	if kill == ord('q'): # when pressed q, closes all
		break

captured.release()
cv2.destroyAllWindows()
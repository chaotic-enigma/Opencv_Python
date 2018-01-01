import cv2
import numpy as np

captured = cv2.VideoCapture(0)

while True:

	_,frame = captured.read()

	gaussian_f = cv2.GaussianBlur(frame,(5,5),0)
	cv2.imshow('gaussian blur',gaussian_f)

	# laplacian edge detection
	laplacian = cv2.Laplacian(gaussian_f,cv2.CV_64F)
	cv2.imshow('laplacian',laplacian)

	# sobel edge detection
	sobel_x = cv2.Sobel(gaussian_f,cv2.CV_64F,1,0,ksize=5)
	cv2.imshow('sobel_x',sobel_x)

	sobel_y = cv2.Sobel(gaussian_f,cv2.CV_64F,0,1,ksize=5)
	cv2.imshow('sobel_y',sobel_y)

	sobel_xy = cv2.addWeighted(sobel_x,0.6,sobel_y,0.4,0)
	cv2.imshow('sobel_xy',sobel_xy)

	# canny edge detection
	canny = cv2.Canny(gaussian_f,100,100)
	cv2.imshow('canny',canny)

	kill = cv2.waitKey(1) & 0xFF
	if kill == ord('q'):
		break

captured.release()
cv2.destroyAllWindows()

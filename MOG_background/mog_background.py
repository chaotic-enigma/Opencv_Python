import cv2
import numpy as np

capture = cv2.VideoCapture('people-walking.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2() # background subtraction

while True:
    ret,frame = capture.read()
    fgmask = fgbg.apply(frame)

    cv2.imshow('original',frame)
    cv2.imshow('fgbg',fgmask)

    k = cv2.waitKey(30) & 0xff

    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()

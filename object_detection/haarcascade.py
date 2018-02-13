import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

capture = cv2.VideoCapture(0)

while True:
    ret,frame = capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5) # depending upon the image and the liklihood that we use

    for (x,y,w,h) in faces:
        cv2.rectangle((frame),(x,y),(x+w,y+h),(255,0,0),2) # ((where),(starting point),(ending point),(color),line_width)
        # eye location
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        # eyes inside face (loop inside loop)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle((roi_color),(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('object_detection',frame)
    k = cv2.waitKey(30) & 0xff

    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()


# video breaks down into frames, frames break down into impages

import cv2
import numpy as np

captured = cv2.VideoCapture(0) # 0 - first webcam in my system
										 # 1 - second webcam in my system ...

# save video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc,25.0,(640,480)) # (output_title,file_name,play_time,(size))

while True: # infinite frames

	ret,frame = captured.read() # read the frames continuously
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # convert color into gray scale

	out.write(frame) # write video
	#print out

	#cv2.imshow('video',frame)
	cv2.imshow('gray_video',gray)

	if cv2.waitKey(1) & 0xFF == ord('q'): # stop the capturing by pressing q
		#print ret
		#print frame
		break

captured.release() # releases the capture so the camera
out.release() # release out

cv2.destroyAllWindows()
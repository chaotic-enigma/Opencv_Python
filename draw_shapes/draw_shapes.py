import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('fedora.png',cv2.IMREAD_COLOR)

# cv2 color format is BGR
# matplotlib color format is RGB

cv2.line(img,(0,0),(350,350),(255,255,255),3) # (where,(start),(end),(color),linewidth)
cv2.rectangle(img,(425,15),(300,50),(0,255,0),3) # (where,(random_point),(random_point),(color),linewidth) joins two poins by 4 lines
cv2.circle(img,(100,450),50,(0,0,255),3) # (where,(center),radius,(color),linewidth)
cv2.circle(img,(234,456),40,(0,0,255),-1) # -1 fills the circle

pts = np.array([[130,43],[23,165],[155,667],[605,487],[430,420]],np.int32)
pts = pts.reshape(-1,1,2)
cv2.polylines(img,[pts],True,(255,0,0),3) # (where,[points],(optional -> if True joins the last point to the first point else, will not join),color,width)

# put text on image
font = cv2.FONT_HERSHEY_SIMPLEX # take font
cv2.putText(img,'Computer Vision',(550,130),font,1.5,(96,96,96),3,cv2.LINE_AA) # (where,text,(start),font_style,font_size,(color),thickness,Anti-Aliasing)

cv2.imwrite('cv2_fedora.png',img)

cv2.imshow('fedora',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.savefig('plt_fedora.png')
plt.show()
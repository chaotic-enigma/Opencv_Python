import cv2
import numpy as np
import matplotlib.pyplot as plt

feature_template = cv2.imread('feature_template.jpg')
features_img = cv2.imread('feature_image.jpg')

orb = cv2.ORB_create() # detector of similarities

# key points  and descriptors
kp1,des1 = orb.detectAndCompute(feature_template,None)
kp2,des2 = orb.detectAndCompute(features_img,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

# find matches and sort them
matches = bf.match(des1,des2)
matches = sorted(matches,key=lambda x:x.distance)

# output images
output_image = cv2.drawMatches(feature_template,kp1,features_img,kp2,matches[:30],None,flags=2)

cv2.imshow('output_image',output_image)

cv2.imwrite('homography_features_match.jpg',output_image)

cv2.waitKey()
cv2.destroyAllWindows()

plt.imshow(output_image)
plt.xticks([],[])
plt.yticks([],[])
plt.show()

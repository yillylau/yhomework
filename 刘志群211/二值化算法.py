import cv2
import numpy as np

img = cv2.imread("E:/lenna.png")
h,w = img.shape[:2]                              
img_gray = np.zeros([h,w],img.dtype)           
for i in range(h):
    for j in range(w):
        m = img[i,j]                             
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3) 

img_binary = np.where(img_gray/255.0 >= 0.5, 1., 0.) 

cv2.imshow('lenna',img)
cv2.imshow("img_binary",img_binary)
cv2.waitKey()
cv2.destroyAllWindows()

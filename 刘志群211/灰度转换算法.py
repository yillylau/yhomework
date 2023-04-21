import cv2
import numpy as np

img = cv2.imread("E:/lenna.png")
height,width,channels =img.shape                           #获取图片的high和wide
img_gray = np.zeros([height,width],img.dtype)                   #创建一张和当前图片大小一样的单通道图片
for i in range(height):
    for j in range(width):
        m = img[i,j]                             
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3) 

img_binary = np.where(img_gray/255.0 >= 0.5, 1., 0.) 

cv2.imshow('lenna',img)
cv2.imshow("img_binary",img_binary)
cv2.waitKey()

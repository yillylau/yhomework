import cv2
import numpy as np
from matplotlib import pyplot as plt

image=cv2.imread("E:\lenna.png")

b,g,r=cv2.split(image)
gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

dst1=cv2.equalizeHist(gray)
dst2=cv2.equalizeHist(b)
dst3=cv2.equalizeHist(g)
dst4=cv2.equalizeHist(r)
dst=cv2.merge((dst2,dst3,dst4))


cv2.imshow("1",dst1)
cv2.imshow("2",dst)


plt.subplot(231)
plt.hist(dst.ravel(),256)
plt.subplot(232)
plt.hist(dst1.ravel(),256)

plt.show()

cv2.waitKey(0)
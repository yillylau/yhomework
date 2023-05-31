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

'''
plt.hist(dst2.ravel(),256)
plt.hist(dst3.ravel(),256)
plt.hist(dst4.ravel(),256)
'''
plt.hist(dst1.ravel(),256)
plt.hist(dst.ravel(),256)
plt.show()

'''
用cv2计算直方图
hist1= cv2.calcHist([dst1],[0],None,[256],[0,255])
hist2= cv2.calcHist([dst2],[0],None,[256],[0,255])
hist3= cv2.calcHist([dst3],[0],None,[256],[0,255])
hist4= cv2.calcHist([dst4],[0],None,[256],[0,255])
hist=cv2.calcHist([dst],[0],None,[256],[0,255])
用cv2一定要有个cv2的返回值承接
cv2.imshow("1",image)
cv2没有直接画图的函数,需要借助plt函数
plt.plot(hist1)
plt.plot(hist2)
plt.plot(hist3)
plt.plot(hist4)
plt.plot(hist)
plt.show()
'''
cv2.waitKey(0)
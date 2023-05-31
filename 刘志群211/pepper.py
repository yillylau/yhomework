import numpy as np
import cv2
from numpy import shape
import random
def fun1(src,percentage):
    NoiseImg=src
    NoiseNum=int(percentage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        if random.random()<=0.5:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255
    return NoiseImg

img=cv2.imread('E:/lenna.png',0)
img1=fun1(img,0.1)
cv2.imshow('pepper',img1)
cv2.waitKey(0)
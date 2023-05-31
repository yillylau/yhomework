import numpy as np
import cv2

img = cv2.imread("E:/lenna.png")
def function(img,out_dim):
    height,width,channels =img.shape
    new_h,new_w=800,800
    scale_x,scale_y = new_w /width,new_h/height
    if new_h == height and new_w == width:
        return img.copy()

    new_img = np.zeros((new_h,new_w,3),dtype=np.uint8)

    for i in range(3):
        for j in range(new_h):
            for k in range(new_w):
                m,n=k/(scale_x+0.5)-0.5,j/(scale_y+0.5)-0.5
                x1,y1=int(k/(scale_x+0.5)-0.5),int(j/(scale_y+0.5)-0.5)#原图中能对应到放大图中的像素点
       
                x2=min(800,x1+1)
                y2=min(800,y1+1)
        
                temp0=(x2-m)*img[y1,x1,i]+(m-x1)*img[y1,x2,i]#y=y1时候的插值点
                temp1=(x2-m)*img[y2,x1,i]+(m-x1)*img[y2,x2,i]#y=y2时候的插值点

                new_img[j,k,i]=int((y2-n)*temp0+(n-y1)*temp1)
    return new_img
                
new_img=function(img,800)
cv2.imshow('bilinear interp',new_img)
cv2.waitKey(0)


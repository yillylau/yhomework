import cv2

img = cv2.imread("E:/lenna.png")
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

cv2.imshow('Gray Image', gray_img)
cv2.imshow('lenna',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
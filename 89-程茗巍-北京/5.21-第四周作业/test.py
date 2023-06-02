from PIL import Image
import numpy as np

filename = 'lenna.png'
im = Image.open(filename)
data = list(im.getdata())
width, height = im.size
data1 =  np.array(data).reshape(width, height,3)
print(data1[1][0])
for index,i in enumerate(data1[1][0]):
    print(i)
    if i > 225:
        print(data1[1][0][index])
        data1[1][0][index] = 1
print(data1[1][0])
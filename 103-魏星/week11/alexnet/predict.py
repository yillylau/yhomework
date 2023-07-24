import numpy as np
import cv2
import torch

from AlexNet import AlexNet

if __name__ == "__main__":
    print("torch mnist begin")
    model = AlexNet(num_classes=10)
    model.load_state_dict(torch.load("./torch-AlexNet.h5"))

    img1 = cv2.imread("./Test.jpg")
    img1 = cv2.resize(img1,(224,224))
    outputs = model.forward(torch.tensor(np.float32(img1.reshape(-1, 3, 224, 224))))
    _, img1_result = torch.max(outputs.data, 1)
    print("模型预测结果是：", img1_result.item())

    print("torch mnist end")
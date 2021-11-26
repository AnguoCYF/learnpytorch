import torch
from torch import nn


input_image = torch.rand(3, 28, 28)
print(input_image.size())

# Flatten曾将每个28*28的二维图像转换成784个像素值的连续数组（minibatch的维度(dim=0)被保持）
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# 线性层用于存储权重和偏置对输入进行线性变换
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# 非线性激活是再模型的输入和输出之间建立复杂的映射关系，以此引入非线性，帮助神经网络学习各种各样的函数。
print(f"before Relu:{hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"after relu:{hidden1}")

# Sequential是一个有序模块的容器，数据以定义的顺寻通过所有模块。
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)

)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

print("Model structure", model, "\n\n")

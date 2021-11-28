import torch
import torch.onnx as onnx
import torchvision.models as models

# Pytorch模型将学习到的参数存储在内部状态字典中，称为state_dict。这些可以通过torch.save方法持久化
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

# 为了记载模型的权重，你需要先创建一个相同模型的实例，然后用load_state_dict()方法加载参数。
model = models.vgg16()
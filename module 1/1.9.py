import torch
import torch.onnx as onnx
import torchvision.models as models

# Pytorch模型将学习到的参数存储在内部状态字典中，称为state_dict。这些可以通过torch.save方法持久化
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

# 为了记载模型的权重，你需要先创建一个相同模型的实例，然后用load_state_dict()方法加载参数。
model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
# 请确保在推理前调用model.eval()方法，以将dropout和batch normalization层设置为eval模式，若不如此做，将产生不一样的推理结果。


# 保存和载入模型的形状，加载模型权重前，需要先将模型实例化
torch.save(model, 'model.pth')
model = torch.load('model.pth')

# 将模型导出为ONNX，由于pytorch执行图的动态性质，导出过程必须遍历执行图以产生持久的ONNX模型：向导出程序传递一个适当大小的测试变量。
input_image = torch.zeros((1, 3, 224, 224))
onnx.export(model, input_image, 'model.onnx')



import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b

print(z.requires_grad)

#   禁用梯度跟踪的原因：
#   1.将神经网络中的一些参数标记为冻结参数。这是对预训练的网络进行微调的一个非常常见的情况。
#   2.当你制作前向传递是，为了加快运算速度，对不跟踪梯度的tensor的计算会更有效率。
with torch.no_grad():  # 法1： 用该方法包围计算代码来停止跟踪
    z = torch.matmul(x, w) + b
print(z.requires_grad)

z_det = z.detach()
print(z_det.requires_grad)  # 法2： 在tensor上使用detach()方法

loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# print('Gradient function for z = ', z.grad_fn)
# print('Gradient function for loss =', loss.grad_fn)

# 当调用.backwoard()时，后向传递开始了。
# 1.计算每个.grad_fn的梯度；
# 2.将它们累计到各自tensor的.grad属性中；
# 3.使用链式规则，一直传播到叶子tensor。
# 注意： 在PYTORCH中，图是从头开始重新创建的；在每次调用.backward()后，autograd开始填充一个新的图
# 这正式允许你在模型中使用控制流语句的原因；如果需要，你可以在每次迭代时改变形状，大小和操作。
# loss.backward()
# print(w.grad)
# print(b.grad)

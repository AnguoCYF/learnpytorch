import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b

print(z.requires_grad)

#   禁用梯度跟踪的原因：
#   1.将神经网络中的一些参数标记为冻结参数。这是对预训练的网络进行微调的一个非常常见的情况。
#   2.当你制作前向传递是，为了加快运算速度，对不跟踪梯度的tensor的计算会更有效率。
with torch.no_grad():   # 法1： 用该方法包围计算代码来停止跟踪
    z = torch.matmul(x, w)+b
print(z.requires_grad)

z_det = z.detach()
print(z_det.requires_grad)  # 法2： 在tensor上使用detach()方法

loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# print('Gradient function for z = ', z.grad_fn)
# print('Gradient function for loss =', loss.grad_fn)

# loss.backward()
# print(w.grad)
# print(b.grad)


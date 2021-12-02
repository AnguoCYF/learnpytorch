import torch, torchvision

model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

prediction = model(data)  # forward pass
loss = (prediction - labels).sum()
loss.backward()  # backward pass

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step()  # gradient descent

# Autograd
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2

external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

# print(9*a**2 == a.grad)
# print(-2*b == b.grad)

# torch.autograd追踪所有将require_grad标志设置为True的tensor上的操作。对于不需要梯度的tensor，将此属性设置为Flase将其排除子啊梯度计算DAG外。
x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)
a = x + y   # 即使只有一个输入tensor的requires_grad=True,一个操作的输出tensor也需要梯度。
print(f"dose 'a' requires gradients? : {a.requires_grad}")
b = x + z
print(f"dose 'b' requires gradients? : {b.requires_grad}")


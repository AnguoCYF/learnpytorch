import torch
import numpy as np

# 直接初始化
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# 从Numpy中创建tensor
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 从其他tensor中初始化，新的tensor保留原tensor的属性（形状，数据类型），除非明确重写
x_ones = torch.ones_like(x_data)
# print(f"ones tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
# print(f"random tensor: \n {x_rand} \n")

# shape用法
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# print(f"random tensor: \n {rand_tensor} \n")
# print(f"ones tensor: \n {ones_tensor} \n")
# print(f"zeros tensor: \n {zeros_tensor} \n")

# tensor属性描述了形状，数据类型及存储设备
# tensor = torch.rand(3, 4)
# print(tensor.shape, tensor.dtype, tensor.device)

# tensor切片
tensor = torch.ones(4, 4)
# print('first row', tensor[0])
# print('first column', tensor[:, 0])
# print('last column', tensor[..., -1])
tensor[:, 1] = 0
# print(tensor)

# tensor.cat 实例
t1 = torch.cat([tensor, tensor, tensor], dim=1)
# print(t1)

# 算术运算
# 矩阵乘法 X.T代表矩阵X的转置
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
# print(y1, y2, y3)

# 元素相乘
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
# print(z1, z2, z3)

# 单元素tensor加和
agg = tensor.sum()
agg_item = agg.item()
# print(agg, agg_item, type(agg_item))

# 原位操作 将结果存储到操作数中的操作被称为原位操作。它们用后缀表示。 例如：x.copy(y), x.t_()
# print(tensor, "\n")
tensor.add_(5)
# print(tensor)

# numpy和tensor可以共享底层内存位置
t = torch.ones(5)
# print(f"t: {t}")
n = t.numpy()
# print(f"n: {n}")

t.add_(1)
# print(f"t:{t}")
# print(f"n: {n}")

# numpy数组转化为tensor
n = np.ones(5)
t = torch.from_numpy(n)
# print(f"t:{t}")
# print(f"n: {n}")

np.add(n, 1, out=n)
print(f"t:{t}")
print(f"n: {n}")
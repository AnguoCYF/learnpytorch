import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()

# 超参数
# 1、epoch数 - 在数据集上迭代的次数
# 2、批量大小 - 在更新参数之前，通过网络传播的数据样本的数量
# 3、学习率
learning_rate = 1e-3
batch_size = 64
epochs = 10

# 优化Loop
# 通过优化Loop来训练和优化模型，优化循环的每一次迭代称为一个epoch。
# 每个epoch主要由两部分构成
# 1、训练Loop -- 在训练集上迭代，试图收敛到最佳参数。
# 2、验证/测试循环 -- 迭代测试数据集，以检查模型性能是否提高。

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器: 优化实在每个训练步骤中调整模型参数以减少模型误差的过程
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 训练LOOP中，优化三步进行。
# 1、调用optimizer.zero_grad()来重置模型参数的梯度。梯度默认为累加；为了防止重复计算我们在每次迭代中明确地将其归零。
# 2、通过调用loss.backwards()对预测损失进行反向传播。Pytorch将损失的梯度与每个参数联系在一起。
# 3、一旦我们有了梯度，我们就可以调用optimizer.step()来根据向后传递中收集的梯度调整参数。


# train_loop负责循环我们的优化代码，test_loop根据测试数据评估模型的性能。
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss:{loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \ n")


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

from torch import nn, optim
import torchvision

model = torchvision.models.resnet18(pretrained=True)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, 10)
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)


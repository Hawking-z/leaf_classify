import torch
import torchvision
from torchinfo import summary
print(torchvision.models.resnet18())
summary(torchvision.models.resnet18(), input_size=(1, 3, 224, 224), device='cpu')


class MyResNet(torch.nn.Module):
    def __init__(self):
        super(MyResNet, self).__init__()
        self.resnet = torchvision.models.resnet18()
        self.resnet.fc = torch.nn.Linear(in_features=512, out_features=10)
    
    def forward(self, x):
        return self.resnet(x)
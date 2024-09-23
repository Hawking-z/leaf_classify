import torch
from torch import nn
from torch.nn import functional as F
import torchvision


class Residual(nn.Module):  
    def __init__(self, input_channels, out_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
       
        self.conv1 = nn.Conv2d(input_channels, out_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, out_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)



class MyResNet18(nn.Module):
    def __init__(self,class_num,pretrained=False):
        super(MyResNet18, self).__init__()
        if pretrained:
            self.resnet = torchvision.models.resnet18(pretrained=True)
            self.resnet.fc = torch.nn.Linear(in_features=512, out_features=class_num)
            
        else:
            self.stage1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            self.stage2 = nn.Sequential(
                Residual(64, 64),
                Residual(64, 64)
            )
            self.stage3 = nn.Sequential(
                Residual(64, 128, use_1x1conv=True, strides=2),
                Residual(128, 128)
            )
            self.stage4 = nn.Sequential(
                Residual(128, 256, use_1x1conv=True, strides=2),
                Residual(256, 256)
            )
            self.stage5 = nn.Sequential(
                Residual(256, 512, use_1x1conv=True, strides=2),
                Residual(512, 512)
            )
            self.fc = nn.Linear(512, class_num)
            self.resnet = nn.Sequential(self.stage1,self.stage2,self.stage3,self.stage4,self.stage5,
                                        nn.AdaptiveAvgPool2d((1, 1)),nn.Flatten(),self.fc)
    
    def forward(self, x):
        return self.resnet(x)



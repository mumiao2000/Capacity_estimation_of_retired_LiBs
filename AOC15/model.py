import torch
import torch.nn as nn
import torchvision
import math

class CNN(nn.Module):
    def __init__(self, seq_len=256, hw_rate=1):
        super().__init__()
        self.width = int(math.sqrt(seq_len/hw_rate)) + 1
        self.height = int(self.width * hw_rate) + 1
        self.pool = nn.AdaptiveAvgPool2d((2, self.height * self.width))
        self.resnet = torchvision.models.resnet18()
        self.resnet.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # remove layer3 & layer4
        self.resnet.layer3 = nn.Sequential()
        self.resnet.layer4 = nn.Sequential()
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid())

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = x.reshape(-1, 2, self.height, self.width)
        return self.resnet(x)


class MyCustomLoss(nn.Module):
    def __init__(self, Y_min, Y_max):
        self.eps = Y_min.item() / (Y_max.item() - Y_min.item())
        super().__init__()

    def forward(self, output, Y):
        rmse_loss = torch.sqrt(nn.MSELoss()(output, Y))
        mae_loss = torch.mean(torch.abs(output - Y))
        mape_loss = torch.mean(torch.abs((output - Y) / (Y + self.eps)))
        return rmse_loss, mae_loss, mape_loss
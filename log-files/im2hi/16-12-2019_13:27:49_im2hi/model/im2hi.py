import torch, torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchsummary import summary
import numpy as np


class IM2HI(nn.Module):
    def __init__(self):
        super(IM2HI, self).__init__()

        self.block_pool_1 = nn.Sequential(
            ResidualBlock(3, 64),
            nn.MaxPool2d(2, 2, return_indices=True)
        )

        self.block_pool_2 = nn.Sequential(
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, 2, return_indices=True)
        )

        self.block_pool_3 = nn.Sequential(
            ResidualBlock(128, 256),
            nn.MaxPool2d(2, 2, return_indices=True)
        )

        self.block_pool_4 = nn.Sequential(
            ResidualBlock(256, 512),
            nn.MaxPool2d(2, 2, return_indices=True)
        )

        self.block_5 = ResidualBlock(512, 256)
        self.block_6 = ResidualBlock(256, 128)
        self.block_7 = ResidualBlock(128, 64)
        self.block_8 = ResidualBlock(128, 1)

    def forward(self, x):
        identity = self.block_pool_1[0](x)
        x, idx_1 = self.block_pool_1(x)
        x, idx_2 = self.block_pool_2(x)
        x, idx_3 = self.block_pool_3(x)
        x, idx_4 = self.block_pool_4(x)

        x = self.block_5(F.max_unpool2d(x, idx_4, 2, stride=2))
        x = self.block_6(F.max_unpool2d(x, idx_3, 2, stride=2))
        x = self.block_7(F.max_unpool2d(x, idx_2, 2, stride=2))

        x = F.max_unpool2d(x, idx_1, 2, stride=2)
        x = torch.cat((identity, x), 1)
        x = self.block_8(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()

        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.features = nn.Sequential(
            # nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_channels, affine=False),
            # nn.ReLU(),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        residual = self.residual(inputs)
        x = self.features(residual)
        return self.relu(residual + x)


if __name__ == '__main__':
    x = torch.rand(1,3,256,256)

    model = IM2HI()
    out = model(x)
    print(out.shape)
    # print(out)
    summary(model, (3,256,256))

    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print(params)
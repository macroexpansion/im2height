import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Yolo3Tiny(nn.Module):
    def __init__(self):
        super(Yolo3Tiny, self).__init__()

        self.features = nn.Sequential(DBL(3, 16, 3, stride=1, padding=1),
                                      nn.MaxPool2d(2, stride=2), #
                                      DBL(16, 32, 3, stride=1, padding=1),
                                      nn.MaxPool2d(2, stride=2), #
                                      DBL(32, 64, 3, stride=1, padding=1),
                                      nn.MaxPool2d(2, stride=2), #
                                      DBL(64, 128, 3, stride=1, padding=1),
                                      nn.MaxPool2d(2, stride=2), #
                                      DBL(128, 256, 3, stride=1, padding=1))

        self.scale1 = nn.Sequential(nn.MaxPool2d(2, stride=2),
                                    DBL(256, 512, 3, stride=1, padding=1),
                                    nn.ConstantPad2d((0,1,0,1), 0),
                                    nn.MaxPool2d(2, stride=1),
                                    DBL(512, 1024, 3, stride=1, padding=1),
                                    DBL(1024, 256, 1, padding=0))

        self.conv1 = nn.Sequential(DBL(256, 128, 1, padding=0))

        self.conv2 = nn.Sequential(DBL(384, 256, 3),
                                   DBL(256, 69, 1, padding=0))

        self.scale2 = nn.Sequential(DBL(256, 512, 3, padding=1),
                                    DBL(512, 69, 1, padding=0))


    def forward(self, inputs):
        feature = self.features(inputs)

        x = self.scale1(feature)

        scale_1 = self.conv1(x)
        scale_1 = F.interpolate(scale_1, scale_factor=(2, 2))
        scale_1 = torch.cat((scale_1, feature), 1)
        scale_1 = self.conv2(scale_1)

        scale_2 = self.scale2(x)

        return scale_1, scale_2 


class DBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, name=''):
        super(DBL, self).__init__()

        self.in_features = in_channels
        self.out_features = out_channels

        self.conv = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.leakyrelu = nn.LeakyReLU(0.1)


    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.leakyrelu(x)
        return x


    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


if __name__ == '__main__':
    test = torch.randn( 2, 2)
    # print(test)
    model = Yolo3Tiny()
    summary(model, (3,416,416))
    # for name, param in model.named_parameters():
        # if param.requires_grad:
            # print(name)
            # print(param.data.shape)
            # break
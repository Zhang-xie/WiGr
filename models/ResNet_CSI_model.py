import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from torchsummary import summary
import pytorch_lightning as pl


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_CSI(nn.Module):

    def __init__(self, block, layers, strides,  inchannel, groups):
        # block can choose basicblock or bottleneck ; layers means the layer1,layer2,layer3,layer4
        # (how many sublayers to stack); inchannel is the input channels ; activity_num is the categories; groups is
        # conv group
        super(ResNet_CSI, self).__init__()
        self.groups = groups
        self.inplanes = 128 * self.groups
        self.conv1 = nn.Conv1d(inchannel, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False, groups=self.groups)

        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        layer = []
        for i, element in enumerate(layers):
            layer.append(self._make_layer(block, self.inplanes * layers[i], element, stride=strides[i]))
        self.layer = nn.Sequential(*layer)

        self.feature = nn.Sequential(
            nn.Conv1d(self.inplanes * block.expansion, self.inplanes*2 * block.expansion, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.inplanes* 2 * block.expansion),
            nn.ReLU(inplace=True),
        )
        self.out_dim = self.inplanes*2 * block.expansion  # self.inplanes*2 * block.expansion

    def _make_layer(self, block, planes, num_layer, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_layer):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c = self.layer(x)
        feaure = torch.squeeze(self.feature(c))
        return feaure


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    model = ResNet_CSI(block=BasicBlock, layers=[1,1,1,1],strides=[2,2,2,2], inchannel=342,groups=3).to(device)
    summary(model, (342,1800))

    # model = ResNet_CSI(block=BasicBlock, layers=[1, 1, 1, 1],strides=[1,1,2,2], inchannel=180, activity_num=150, groups=3).to(device)
    # summary(model, (180, 200))

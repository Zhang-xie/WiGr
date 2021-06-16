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


class ARIL(nn.Module):
    def __init__(self, block, layers,  inchannel=52, activity_num=6, location_num=16):
        super(ARIL, self).__init__()
        self.inplanes = 128
        self.conv1 = nn.Conv1d(inchannel, 128, kernel_size=7, stride=2, padding=3,
                                 bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.ACTClassifier = nn.Sequential(
            nn.Conv1d(128 * block.expansion, 128 * block.expansion, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(128 * block.expansion),
            nn.ReLU(inplace=True),
            # nn.Conv1d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=0, bias=False),
            # nn.BatchNorm1d(512 * block.expansion),
            # nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.act_fc = nn.Linear(128 * block.expansion, activity_num)

        # self.LOCClassifier = nn.Sequential(
        #     nn.Conv1d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=0, bias=False),
        #     nn.BatchNorm1d(512 * block.expansion),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool1d(1),
        # )
        # self.loc_fc = nn.Linear(512 * block.expansion, location_num)
        # self.loc_fc_f = nn.Linear(256, location_num)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,L = y.size()
        return F.interpolate(x, size=L) + y

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        act = self.ACTClassifier(c4)
        act = act.view(act.size(0), -1)
        act1 = self.act_fc(act)
        # loc = self.LOCClassifier(c4)
        # loc = loc.view(loc.size(0), -1)
        # loc1 = self.loc_fc(loc)
        return act1


class ARIL_pl(pl.LightningModule):
    def __init__(self,inchannel=52,layers=None):
        super().__init__()
        if layers is None:
            layers = [1,1,1,1]
        self.aril_model = ARIL(block=BasicBlock, layers=layers, inchannel=inchannel)
        self.criterion = nn.CrossEntropyLoss(size_average=False)
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.acc_record = []

    def forward(self,x):
        return self.aril_model(x)

    def training_step(self, batch, batch_idx):
        activity_label, loc_label, data = batch

        pre_act_label = self(data)

        loss = self.criterion(pre_act_label,activity_label.long().squeeze())

        self.log("GesTr_loss",loss)
        self.train_acc(pre_act_label,activity_label.long().squeeze())

        return loss

    def validation_step(self,batch, batch_idx):
        activity_label, loc_label, data = batch

        pre_act_label = self(data)

        loss = self.criterion(pre_act_label, activity_label.long().squeeze())

        self.log("GesVa_loss", loss)
        self.val_acc(pre_act_label, activity_label.long().squeeze())

        return loss

    def training_epoch_end(self,training_step_outputs):
        self.log('GesTr_Acc', self.train_acc.compute())

    def validation_epoch_end(self,val_step_outputs):
        self.log('GesVa_Acc', self.val_acc.compute())
        self.acc_record.append(self.val_acc.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.aril_model.parameters(), lr=0.005)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[10, 20, 30, 40, 60, 70, 80, 90, 100, 110, 120, 130,
                                                                     140, 150, 160, 170, 180, 190, 200, 250, 300],
                                                         gamma=0.5)
        return [optimizer,], [scheduler,]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    model = ARIL(block=BasicBlock, layers=[1, 1, 1, 1], inchannel=52).to(device)
    summary(model, (52,200))

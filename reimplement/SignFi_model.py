import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from torchsummary import summary
import pytorch_lightning as pl


class SignFi_model(nn.Module):
    def __init__(self,num_class):
        super(SignFi_model,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(3,3,[3,3],[1,1],[1,1]),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            # nn.AvgPool2d([3, 3], [3, 3], [0, 0]),
            # nn.Dropout2d(0.6),
        )
        self.layer2=nn.Sequential(
            nn.AvgPool2d([3, 3], [3, 3], [0, 0]),
            nn.Dropout2d(0.6),
        )
        self.fc=nn.Linear(4000,num_class)

    def forward(self, in_data):
        out = self.layer1(in_data)
        out = self.layer2(out.transpose(1,2))
        out = self.fc(out.reshape(out.shape[0],-1))
        return out


class SignFi_pl(pl.LightningModule):
    def __init__(self):
        super(SignFi_pl, self).__init__()
        self.signfi_model = SignFi_model(150)
        self.criterion = nn.CrossEntropyLoss(size_average=False)
        self.tran_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self,x):
        return self.signfi_model(x)

    def training_step(self, batch, batch_idx):
        activity_label, loc_label, data = batch

        pre_act_label = self(data)

        loss = self.criterion(pre_act_label,activity_label.long().squeeze())

        self.log("train_loss",loss)
        self.tran_acc(pre_act_label,activity_label.long().squeeze())

        return loss

    def test_step(self,batch, batch_idx):
        activity_label, loc_label, data = batch

        pre_act_label = self(data)

        loss = self.criterion(pre_act_label, activity_label.long().squeeze())

        self.log("test_loss", loss)
        self.test_acc(pre_act_label, activity_label.long().squeeze())

        return loss

    def training_epoch_end(self,training_step_outputs):
        self.log('GesClaAcc', self.train_acc.compute())

    def test_step_end(self):
        self.log('test_acc',self.test_acc.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[10, 20, 30, 40, 60, 70, 80, 90, 100, 110, 120, 130,
                                                                     140, 150, 160, 170, 180, 190, 200, 250, 300],
                                                         gamma=0.5)
        return optimizer, scheduler


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    model = SignFi_model(150).to(device)
    summary(model, (3,200,60))
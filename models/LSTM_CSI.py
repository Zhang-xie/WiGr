import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence
import random
import pytorch_lightning as pl
from torchsummary import summary


class RAW_Feature_Encoder(nn.Module):
    def __init__(self, in_c, out_feature_dim):
        super().__init__()
        self.l = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_feature_dim, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        x = self.l(x)
        return x.squeeze()


class Seq_Classifier_RNN(nn.Module):
    def __init__(self, in_feature_dim, lstm_layer, class_nums):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=in_feature_dim, hidden_size=128, num_layers=lstm_layer
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, class_nums)
        )

    def forward(self, input):
        x = pack_sequence(input, enforce_sorted=False)
        output, _ = self.rnn(x)   # (num_layers * num_directions, batch, hidden_size)
        seq_unpacked, lens_unpacked = pad_packed_sequence(output)
        lens_unpacked -= 1
        seq_out = torch.stack(
            [seq_unpacked[lens_unpacked[i], i, :] for i in range(len(lens_unpacked))]
        )
        x = self.decoder(seq_out)
        return x


class CNN_LSTM_CSI(pl.LightningModule):
    def __init__(self,in_channel=3, feature_dim=64,lstm_layer_num=3,num_class=6):
        super().__init__()
        self.in_channel = in_channel
        self.feature_dim = feature_dim
        self.lstm_layer_num = lstm_layer_num
        self.num_class = num_class

        self.CNN_encoder = RAW_Feature_Encoder(in_c=self.in_channel, out_feature_dim=self.feature_dim)
        self.LSTM_classifier = Seq_Classifier_RNN(in_feature_dim=self.feature_dim , lstm_layer=self.lstm_layer_num, class_nums=self.num_class)

        self.criterion = nn.CrossEntropyLoss(size_average=False)
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()

    def training_step(self, batch, batch_idx):
        data, activity_label, loc_label, = batch   # the data is a 5 dimensional tensor: [batch,time,in_channel/receiver,length,width]
        activity_label = torch.stack(activity_label)
        loc_label = torch.stack(loc_label)

        input_data = torch.cat(data)  # [batch*time,in_channel,length,width]
        feature_out = self.CNN_encoder(input_data)
        seq_in = torch.split(feature_out, [len(x) for x in data])  # [batch,time,feature_dim]
        pre_act_label = self.LSTM_classifier(seq_in)

        loss = self.criterion(pre_act_label, activity_label.long().squeeze())

        self.log("GesTr_loss", loss)
        self.train_acc(pre_act_label, activity_label.long().squeeze())

        return loss

    def validation_step(self,  batch, batch_idx):
        data, activity_label, loc_label, = batch  # data is a 3 dimensional tensor [batch,90,total_length]

        activity_label = torch.stack(activity_label)
        loc_label = torch.stack(loc_label)
        # the data is a 5 dimensional tensor: [batch,time,in_channel/receiver,length,width]

        input_data = torch.cat(data) # [batch*time,in_channel,length,width]
        feature_out = self.CNN_encoder(input_data)
        seq_in = torch.split(feature_out, [len(x) for x in data])  # [batch,time,feature_dim]
        pre_act_label = self.LSTM_classifier(seq_in)

        loss = self.criterion(pre_act_label, activity_label.long().squeeze())

        self.log("GesVa_loss", loss)
        self.val_acc(pre_act_label, activity_label.long().squeeze())

        return loss

    def validation_epoch_end(self, val_step_outputs):
        self.log('GesVa_Acc', self.val_acc.compute())
        # self.acc_record.append(self.val_acc.compute())

    def training_epoch_end(self, training_step_outputs):
        self.log('GesTr_Acc', self.train_acc.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[40,  80, 120, 160, 200, 240, 280,320,360,400],
                                                         gamma=0.5)
        return [optimizer, ], [scheduler, ]


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    model = RAW_Feature_Encoder(in_c=3, out_feature_dim=64).to(device)
    summary(model, (3,50,30))
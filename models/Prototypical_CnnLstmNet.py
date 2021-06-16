import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence
import random
import pytorch_lightning as pl
from torchsummary import summary
from evaluation import similarity
from .ResNet_CSI_model import ResNet_CSI,BasicBlock,Bottleneck
from pytorch_lightning.metrics import ConfusionMatrix


class RawCSIEncoder(nn.Module):
    def __init__(self, in_channel_cnn, out_feature_dim_cnn):
        super().__init__()
        self.l = nn.Sequential(
            nn.Conv2d(in_channel_cnn, out_feature_dim_cnn//4, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(out_feature_dim_cnn//4),
            nn.ReLU(),
            nn.Conv2d(out_feature_dim_cnn//4, out_feature_dim_cnn//3, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(out_feature_dim_cnn//3),
            nn.ReLU(),
            nn.Conv2d(out_feature_dim_cnn//3, out_feature_dim_cnn//2, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(out_feature_dim_cnn//2),
            nn.ReLU(),
            nn.Conv2d(out_feature_dim_cnn//2, out_feature_dim_cnn, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_feature_dim_cnn,  out_feature_dim_cnn, 1, 1, 0),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.l(x)
        return x.squeeze()


class SeqFeatureEncoder(nn.Module):
    def __init__(self, in_feature_dim_lstm, num_lstm_layer, out_feature_dim_lstm=128,):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=in_feature_dim_lstm, hidden_size=out_feature_dim_lstm, num_layers=num_lstm_layer
        )

    def forward(self, input):
        x = pack_sequence(input, enforce_sorted=False)
        output, _ = self.rnn(x)   # (num_layers * num_directions, batch, hidden_size)
        seq_unpacked, lens_unpacked = pad_packed_sequence(output)
        lens_unpacked -= 1
        seq_out = torch.stack(
            [seq_unpacked[lens_unpacked[i], i, :] for i in range(len(lens_unpacked))]
        )
        return seq_out


class LinearClassifier(nn.Module):
    def __init__(self,in_feature_dim_linear=128,num_class=6):
        super(LinearClassifier, self).__init__()
        self.decoder = nn.Linear(in_feature_dim_linear, num_class,bias=False)

    def forward(self,input):
        return self.decoder(input)


class PrototypicalCnnLstmNet(pl.LightningModule):
    def __init__(self,in_channel_cnn,out_feature_dim_cnn,out_feature_dim_lstm=52,num_lstm_layer=1,
                 metric_method="Euclidean",k_shot=1,num_class_linear_flag=None,combine=False):
        """
        :param in_channel_cnn: the input channel of CNN
        :param out_feature_dim_cnn: the output feature dimension of CNN
        :param out_feature_dim_lstm: the output feature dimension of LSTM
        :param num_lstm_layer: the number of LSTM layers
        :param metric_method: the method to metric similarity : Euclidean, cosine
        :param k_shot: the number of samples per class in the support set
        :param num_class_linear_flag: the number of classes to classifying and using the linear dual path or not
        :param combine: combine the two path results or not
        """
        super().__init__()
        self.alpha = 0.02

        self.in_channel_cnn = in_channel_cnn
        self.out_feature_dim_cnn = out_feature_dim_cnn
        self.out_feature_dim_lstm = out_feature_dim_lstm
        self.num_lstm_layer = num_lstm_layer

        self.metric_method = metric_method
        self.k_shot = k_shot
        self.combine = combine  # we need to combine the linear or not
        self.num_class_linear_flag = num_class_linear_flag  # only using when we add the linear classifier

        self.CNN_encoder = RawCSIEncoder(in_channel_cnn=self.in_channel_cnn,
                                         out_feature_dim_cnn=self.out_feature_dim_cnn)
        self.LSTM_encoder = SeqFeatureEncoder(in_feature_dim_lstm=self.out_feature_dim_cnn,
                                              out_feature_dim_lstm=self.out_feature_dim_lstm,
                                              num_lstm_layer=self.num_lstm_layer)

        if self.num_class_linear_flag is not None:
            self.linear_classifier = LinearClassifier(in_feature_dim_linear=self.out_feature_dim_lstm,num_class=self.num_class_linear_flag)
            self.train_acc_linear = pl.metrics.Accuracy()
            self.val_acc_linear = pl.metrics.Accuracy()

        self.similarity_metric = similarity.Pair_metric(metric_method=self.metric_method)
        self.similarity = similarity.Pair_metric(metric_method="cosine")
        # for calculating the cosine distance between support set feature and linear layer weights W

        self.criterion = nn.CrossEntropyLoss(size_average=False)
        self.train_acc = pl.metrics.Accuracy()  # the training accuracy of metric classifier
        self.val_acc = pl.metrics.Accuracy()  # the validation accuracy of metric classifier

        self.confmat_linear_all = []  # storage all the confusion matrix of linear classifier
        self.comfmat_metric_all = []  # storage all the confusion matrix of metric classifier

    def training_step(self, batch, batch_idx):
        batch = batch[0]
        query_dataset, support_set = batch

        query_data, query_activity_label, = query_dataset   # the data list: [num_sample1,(in_channel,time).tensor]
        qu_activity_label = torch.stack(query_activity_label)

        support_data, support_activity_label, = support_set  # the data list:[num_sample2,(in_channel,time).tensor]
        su_activity_label = torch.stack(support_activity_label)

        # extracting the features
        qu_data = torch.cat(query_data)  # [batch*time,in_channel,length,width]
        su_data = torch.cat(support_data)  # [batch*time,in_channel,length,width]

        qu_feature_out = self.CNN_encoder(qu_data)
        qu_seq_in = torch.split(qu_feature_out, [len(x) for x in query_data])  # [batch,time,feature_dim]
        qu_feature = self.LSTM_encoder(qu_seq_in)

        su_feature_out = self.CNN_encoder(su_data)
        su_seq_in = torch.split(su_feature_out, [len(x) for x in support_data])  # [batch,time,feature_dim]
        su_feature = self.LSTM_encoder(su_seq_in)

        # if num_class_linear_flag is not None, which means we using the Dual path.
        if self.num_class_linear_flag is not None:
            pre_gesture_linear_qu = self.linear_classifier(qu_feature)
            pre_gesture_linear_su = self.linear_classifier(su_feature)
            pre_gesture_linear = torch.cat([pre_gesture_linear_su,pre_gesture_linear_qu])
            gesture_label = torch.cat([su_activity_label , qu_activity_label])
            linear_classifier_loss = self.criterion(pre_gesture_linear,gesture_label.long().squeeze())
            self.log("GesTr_loss_linear", linear_classifier_loss)
            self.train_acc_linear(pre_gesture_linear, gesture_label.long().squeeze())
        else:
            linear_classifier_loss = 0

        # for few-shot, we using average values of all the support set sample-feature as the final feature.
        if self.k_shot != 1:
            su_feature_temp1 = su_feature.reshape(-1,self.k_shot,su_feature.size()[1])
            su_feature_k_shot = su_feature_temp1.mean(1,keepdim=False)
        else:
            su_feature_k_shot = su_feature

        # combine the dual path knowledge
        if self.combine:
            # su_feature_final = torch.ones_like(su_feature_k_shot)
            # w = self.linear_classifier.decoder.weight.detach()
            # cosine_distance = self.similarity(su_feature_k_shot, w)
            # for i in range(len(w)):
            #     cosine = cosine_distance[i][i]
            #     t = 0
            #     for j in range(len(w)):
            #         t += cosine_distance[i][j]
            #     if cosine / t > self.alpha:
            #         replace_support = cosine * w[i] * (torch.norm(su_feature_k_shot[i], p=2) / torch.norm(w[i], p=2))
            #         su_feature_final[i] = replace_support

            su_feature_final = su_feature_k_shot
            w = self.linear_classifier.decoder.weight
            cosine_distance = self.similarity(w, w)
            zero = torch.zeros_like(cosine_distance)
            constraint_element1 = torch.where(cosine_distance < self.alpha, zero, cosine_distance)
            constraint_element = torch.where(constraint_element1 == 1, zero,
                                             constraint_element1)
            loss_orthogonal_constraint = constraint_element.sum() / 2
            linear_classifier_loss += loss_orthogonal_constraint
        else:
            su_feature_final = su_feature_k_shot

        predict_label = self.similarity_metric(qu_feature,su_feature_final)
        loss = self.criterion(predict_label, qu_activity_label.long().squeeze())
        self.log("GesTr_loss", loss)
        self.train_acc(predict_label, qu_activity_label.long().squeeze())

        loss += linear_classifier_loss
        return loss

    def validation_step(self,  batch, batch_idx):
        batch = batch[0]
        query_dataset, support_set = batch

        query_data, query_activity_label, = query_dataset  # the data list: [num_sample1,(in_channel,time).tensor]
        qu_activity_label = torch.stack(query_activity_label)

        support_data, support_activity_label, = support_set  # the data list:[num_sample2,(in_channel,time).tensor]
        su_activity_label = torch.stack(support_activity_label)

        # extracting the features
        qu_data = torch.cat(query_data)  # [batch*time,in_channel,length,width]
        su_data = torch.cat(support_data)  # [batch*time,in_channel,length,width]

        qu_feature_out = self.CNN_encoder(qu_data)
        qu_seq_in = torch.split(qu_feature_out, [len(x) for x in query_data])  # [batch,time,feature_dim]
        qu_feature = self.LSTM_encoder(qu_seq_in)

        su_feature_out = self.CNN_encoder(su_data)
        su_seq_in = torch.split(su_feature_out, [len(x) for x in support_data])  # [batch,time,feature_dim]
        su_feature = self.LSTM_encoder(su_seq_in)

        # if num_class_linear_flag is not None, which means we using the Dual path.
        if self.num_class_linear_flag is not None:
            pre_gesture_linear_qu = self.linear_classifier(qu_feature)
            pre_gesture_linear_su = self.linear_classifier(su_feature)
            pre_gesture_linear = torch.cat([pre_gesture_linear_su, pre_gesture_linear_qu])
            gesture_label = torch.cat([su_activity_label, qu_activity_label])
            linear_classifier_loss = self.criterion(pre_gesture_linear, gesture_label.long().squeeze())
            self.log("GesVa_loss_linear", linear_classifier_loss)
            self.val_acc_linear(pre_gesture_linear, gesture_label.long().squeeze())
            self.confmat_linear.update(pre_gesture_linear.cpu(), gesture_label.long().squeeze().cpu())
        else:
            linear_classifier_loss = 0

        # for few-shot, we using average values of all the support set sample-feature as the final feature.
        if self.k_shot != 1:
            su_feature_temp1 = su_feature.reshape(-1, self.k_shot, su_feature.size()[1])
            su_feature_k_shot = su_feature_temp1.mean(1, keepdim=False)
        else:
            su_feature_k_shot = su_feature

        # combine the dual path knowledge or add the orthogonal constraint
        if self.combine:
            # su_feature_final = torch.ones_like(su_feature_k_shot)
            # w = self.linear_classifier.decoder.weight.detach()
            # cosine_distance = self.similarity(su_feature_k_shot, w)
            # for i in range(len(w)):
            #     cosine = cosine_distance[i][i]
            #     t = 0
            #     for j in range(len(w)):
            #         t += cosine_distance[i][j]
            #     if cosine / t > self.alpha:
            #         replace_support = cosine * w[i] * (torch.norm(su_feature_k_shot[i], p=2) / torch.norm(w[i], p=2))
            #         su_feature_final[i] = replace_support

            su_feature_final = su_feature_k_shot
            w = self.linear_classifier.decoder.weight
            cosine_distance = self.similarity(w, w)
            zero = torch.zeros_like(cosine_distance)
            # constraint_element = torch.where((cosine_distance < self.alpha) or (cosine_distance == 1), zero, cosine_distance)
            # loss_orthogonal_constraint = constraint_element.sum() / 2
            constraint_element1 = torch.where(cosine_distance < self.alpha, zero, cosine_distance)
            constraint_element = torch.where(constraint_element1 == 1, zero,
                                             constraint_element1)
            loss_orthogonal_constraint = constraint_element.sum() / 2
            linear_classifier_loss += loss_orthogonal_constraint
        else:
            su_feature_final = su_feature_k_shot

        predict_label = self.similarity_metric(qu_feature, su_feature_final)
        loss = self.criterion(predict_label, qu_activity_label.long().squeeze())
        self.log("GesVa_loss", loss)
        self.val_acc(predict_label, qu_activity_label.long().squeeze())
        self.confmat_metric.update(predict_label.cpu(), qu_activity_label.long().squeeze().cpu())

        loss += linear_classifier_loss
        return loss

    def on_validation_epoch_start(self):
        self.confmat_metric = ConfusionMatrix(num_classes=6)
        if self.num_class_linear_flag is not None:
            self.confmat_linear = ConfusionMatrix(num_classes=6)

    def validation_epoch_end(self, val_step_outputs):
        self.log('GesVa_Acc', self.val_acc.compute())
        self.comfmat_metric_all.append(self.confmat_metric.compute())

        if self.num_class_linear_flag is not None:
            self.log('GesVa_Acc_linear', self.val_acc_linear.compute())
            self.confmat_linear_all.append(self.confmat_linear.compute())

    def training_epoch_end(self, training_step_outputs):
        self.log('GesTr_Acc', self.train_acc.compute())
        if self.num_class_linear_flag is not None:
            self.log('train_acc_linear', self.train_acc_linear.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[80, 160, 240, 320,400],
                                                         gamma=0.5)
        return [optimizer, ], [scheduler, ]


if __name__ == '__main__':
    pass

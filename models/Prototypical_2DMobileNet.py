import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence
import random
import pytorch_lightning as pl
from torchsummary import summary
from evaluation import similarity
from .MobiV3_CSI_model import MobileNetV3
from pytorch_lightning.metrics import ConfusionMatrix

# from utils import custom_stack
def custom_stack(x,time_dim=2,time_size=1800):
    out=[]
    if isinstance(x,list):
        slc = [slice(None)] * len(x[0].shape)
        slc[time_dim] = slice(0, time_size)
        r_slc=[1]*len(x[0].shape)
        for i in x:
            if i.shape[time_dim]<time_size:
                r_slc[time_dim]=1+time_size//i.shape[time_dim]
                i=i.repeat(*r_slc)
            if i.shape[time_dim]>time_size:
                out.append(i[slc])
    return torch.stack(out)



class LinearClassifier(nn.Module):
    def __init__(self,in_channel=128,num_class=6):
        super(LinearClassifier, self).__init__()
        self.downsample = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.decoder = nn.Linear(in_channel, num_class, bias=False)

    def forward(self,input):
        x = self.downsample(input)
        return self.decoder(x)


class PrototypicalMobileNet(pl.LightningModule):
    def __init__(self,width_mult=1.0,inchannel=3,align=False,
                 metric_method="Euclidean",k_shot=1,num_class_linear_flag=None,combine=False):
        super(PrototypicalMobileNet, self).__init__()
        self.alpha = 0.02

        self.width_mult = width_mult
        self.inchannel = inchannel
        self.align = align

        self.metric_method = metric_method
        self.k_shot = k_shot
        self.combine = combine  # we need to combine the linear or not
        self.num_class_linear_flag = num_class_linear_flag  # only using when we add the linear classifier

        self.MobileNet_encoder = MobileNetV3(width_mult= self.width_mult, inchannel= self.inchannel)  # output: [out_channel, height, width]
        self.feature_dim = self.MobileNet_encoder.final_out_channel

        if self.num_class_linear_flag is not None:
            self.linear_classifier = LinearClassifier(in_channel=self.feature_dim, num_class=self.num_class_linear_flag)
            self.train_acc_linear = pl.metrics.Accuracy()
            self.val_acc_linear = pl.metrics.Accuracy()

        self.similarity_metric = similarity.Pair_metric(metric_method=self.metric_method,inchannel=self.feature_dim * 2, factor=self.width_mult)
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

        query_data, query_activity_label, = query_dataset   # the data list: [num_sample1,(in_channel,subcarries,time).tensor]
        qu_activity_label = torch.stack(query_activity_label)

        support_data, support_activity_label, = support_set  # the data list:[num_sample2,(in_channel,subcarries,time).tensor]
        su_activity_label = torch.stack(support_activity_label)

        # extracting the features
        if self.align:
            qu_data = torch.stack(query_data)  # [num_sample1,in_channel,subcarries,time]
            su_data = torch.stack(support_data)  # [num_sample2,in_channel,subcarries,time]
            qu_feature = self.MobileNet_encoder(qu_data)
            su_feature = self.MobileNet_encoder(su_data)
        else:
            qu_data = custom_stack(query_data,time_dim=2)  # [num_sample1,in_channel,time]
            su_data = custom_stack(support_data,time_dim=2)  # [num_sample2,in_channel,time]
            qu_feature = self.ResNet_encoder(qu_data)
            su_feature = self.ResNet_encoder(su_data)
            # qu_feature_temp = []
            # su_feature_temp = []
            # for i,x in enumerate(query_data):
            #     feature = self.MobileNet_encoder(x)
            #     qu_feature_temp.append(feature)
            # for j,x in enumerate(support_data):
            #     feature = self.MobileNet_encoder(x)
            #     su_feature_temp.append(feature)
            # qu_feature = torch.stack(qu_feature_temp)  # [num_sample1,out_channel,height,width]
            # su_feature = torch.stack(su_feature_temp)  # [num_sample2,out_channel,height,width]

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
            su_feature_temp1 = su_feature.reshape(-1,self.k_shot,su_feature.size()[1],su_feature.size()[2],su_feature.size()[3])
            su_feature_k_shot = su_feature_temp1.mean(1,keepdim=False)
        else:
            su_feature_k_shot = su_feature

        # combine the dual path knowledge
        if self.combine:
            su_feature_final = su_feature_k_shot
            w = self.linear_classifier.decoder.weight
            cosine_distance = self.similarity(w, w)
            zero = torch.zeros_like(cosine_distance)
            # constraint_element = torch.where((cosine_distance < self.alpha) or (cosine_distance == 1), zero, cosine_distance)
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

        query_data, query_activity_label, = query_dataset  # the data list: [num_sample1,(in_channel,,subcarries,time).tensor]
        qu_activity_label = torch.stack(query_activity_label)

        support_data, support_activity_label, = support_set  # the data list:[num_sample2,(in_channel,,subcarries,time).tensor]
        su_activity_label = torch.stack(support_activity_label)

        # extracting the features
        if self.align:
            qu_data = torch.stack(query_data)  # [num_sample1,time,in_channel,,subcarries,time]
            su_data = torch.stack(support_data)  # [num_sample2,time,in_channel,,subcarries,time]
            qu_feature = self.MobileNet_encoder(qu_data)
            su_feature = self.MobileNet_encoder(su_data)
        else:
            qu_data = custom_stack(query_data,time_dim=2)  # [num_sample1,in_channel,time]
            su_data = custom_stack(support_data,time_dim=2)  # [num_sample2,in_channel,time]
            qu_feature = self.ResNet_encoder(qu_data)
            su_feature = self.ResNet_encoder(su_data)
            # qu_feature_temp = []
            # su_feature_temp = []
            # for i, x in enumerate(query_data):
            #     feature = self.MobileNet_encoder(x)
            #     qu_feature_temp.append(feature)
            # for j, x in enumerate(support_data):
            #     feature = self.MobileNet_encoder(x)
            #     su_feature_temp.append(feature)
            # qu_feature = torch.stack(qu_feature_temp)
            # su_feature = torch.stack(su_feature_temp)

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
            su_feature_temp1 = su_feature.reshape(-1, self.k_shot, su_feature.size()[1] , su_feature.size()[2],su_feature.size()[3])
            su_feature_k_shot = su_feature_temp1.mean(1, keepdim=False)

        else:
            su_feature_k_shot = su_feature

        # combine the dual path knowledge or add the orthogonal constraint
        if self.combine:
            su_feature_final = su_feature_k_shot
            w = self.linear_classifier.decoder.weight
            cosine_distance = self.similarity(w, w)
            zero = torch.zeros_like(cosine_distance)
            # constraint_element = torch.where((cosine_distance < self.alpha) or (cosine_distance == 1), zero, cosine_distance)
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

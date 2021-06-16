import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence
import random
import pytorch_lightning as pl
from torchsummary import summary
from evaluation import similarity
from ResNet_CSI_model import ResNet_CSI,BasicBlock,Bottleneck
from pytorch_lightning.metrics import ConfusionMatrix
from torchsummary import summary
import time
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0


def custom_stack(x,time_dim=2,time_size=1800):
    out = []
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
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.decoder = nn.Linear(in_channel, num_class, bias=False)

    def forward(self,input):
        x = self.downsample(input)
        return self.decoder(x)


class PrototypicalResNet(nn.Module):
    def __init__(self,layers,strides,inchannel=52,groups=1,align=False,
                 metric_method="Euclidean",k_shot=1,num_class_linear_flag=None,combine=False):
        """
        :param layers: this is a list, which define the number of types of layers
        :param strides:  the convolution strides of layers
        :param inchannel: input channel
        :param groups: convolution groups
        :param align: whether the length of input series are the same or not
        :param metric_method: the method to metric similarity : Euclidean, cosine
        :param k_shot: the number of samples per class in the support set
        :param num_class_linear_flag: the number of classes to classifying and using the linear dual path or not
        :param combine: combine the two path results or not
        """
        super().__init__()
        self.alpha = 0.02

        self.layers = layers
        self.strides = strides
        self.inchannel = inchannel
        self.groups = groups
        self.align = align

        self.metric_method = metric_method
        self.k_shot = k_shot
        self.combine = combine  # we need to combine the linear or not
        self.num_class_linear_flag = num_class_linear_flag  # only using when we add the linear classifier

        self.ResNet_encoder = ResNet_CSI(block=BasicBlock, layers=self.layers,strides=self.strides, inchannel=self.inchannel,groups=self.groups) # output shape [feature_dim, length]
        self.feature_dim = self.ResNet_encoder.out_dim

        if self.num_class_linear_flag is not None:
            self.linear_classifier = LinearClassifier(in_channel=self.feature_dim,num_class=self.num_class_linear_flag)
            self.train_acc_linear = pl.metrics.Accuracy()
            self.val_acc_linear = pl.metrics.Accuracy()

        self.similarity_metric = similarity.Pair_metric(metric_method=self.metric_method,inchannel=self.feature_dim * 2)
        self.similarity = similarity.Pair_metric(metric_method="cosine")
        # for calculating the cosine distance between support set feature and linear layer weights W

        self.train_acc = pl.metrics.Accuracy()  # the training accuracy of metric classifier
        self.val_acc = pl.metrics.Accuracy()  # the validation accuracy of metric classifier

        self.confmat_linear_all = []  # storage all the confusion matrix of linear classifier
        self.comfmat_metric_all = []  # storage all the confusion matrix of metric classifier

    def forward(self, batch):
        query_dataset, support_set = batch

        query_data, query_activity_label, = query_dataset   # the data list: [num_sample1,(in_channel,time).tensor]
        qu_activity_label = torch.stack(query_activity_label).to(device)

        support_data, support_activity_label, = support_set  # the data list:[num_sample2,(in_channel,time).tensor]
        su_activity_label = torch.stack(support_activity_label).to(device)

        # extracting the features
        if self.align:
            qu_data = torch.stack(query_data)  # [num_sample1,in_channel,time]
            su_data = torch.stack(support_data)  # [num_sample2,in_channel,time]
            qu_feature = self.ResNet_encoder(qu_data)
            su_feature = self.ResNet_encoder(su_data)
        else:
            qu_data = custom_stack(query_data,time_dim=1)  # [num_sample1,in_channel,time]
            su_data = custom_stack(support_data,time_dim=1)  # [num_sample2,in_channel,time]
            qu_feature = self.ResNet_encoder(qu_data)
            su_feature = self.ResNet_encoder(su_data)

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
            su_feature_temp1 = su_feature.reshape(-1,self.k_shot,su_feature.size()[1],su_feature.size()[2])
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
        return predict_label


    #         loss = self.criterion(predict_label, qu_activity_label.long().squeeze())
    #         self.log("GesTr_loss", loss)
    #         self.train_acc(predict_label, qu_activity_label.long().squeeze())
    #
    #         loss += linear_classifier_loss
    #         return loss


if __name__ == '__main__':

    query_data = []
    query_activity_label = []
    for i in range(6):
        x= torch.randn((90,180),).to(device)
        query_data.append(x)
        l = torch.ones((1)).to(device)
        query_activity_label.append(l)
    query_dataset = (query_data,query_activity_label)

    support_data = []
    support_activity_label = []
    for i in range(6):
        x = torch.randn((90, 180),).to(device)
        support_data.append(x)
        l = torch.ones((1)).to(device)
        support_activity_label.append(l)
    support_set = (support_data, support_activity_label)

    batch = (query_dataset, support_set)
    model = PrototypicalResNet(layers=[1,1,1],strides=[1,2,2],inchannel=90,groups=3,align=False,
                 metric_method="Euclidean",k_shot=1,num_class_linear_flag=None,combine=False).to(device)
    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[80, 160, 240, 320, 400],
                                                     gamma=0.5)

    start_time = time.time()
    predict_label = model(batch)
    qu_activity_label = torch.stack(query_activity_label)
    predict_time = time.time()
    loss = criterion(predict_label, qu_activity_label.long().squeeze())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    end_time = time.time()
    print("training time:",end_time-start_time)
    print("training time a sample:", (end_time - start_time)/12.0)
    print("testing time:", predict_time - start_time)
    print("testing time a sample:", (predict_time - start_time)/12.0)

    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')
    print("the number of parameters:", )
    # summary(model, (90, 1800))

from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchsummary import summary


class Pair_metric(nn.Module):
    def __init__(self, metric_method, inchannel=3, factor = 0.25):
        super(Pair_metric, self).__init__()
        self.metric_method = metric_method
        self.inchannel = inchannel
        self.factor = factor
        if self.metric_method == "relation_1D":
            self.relation_nn = Relation_metric_1D(self.inchannel)
        elif self.metric_method == 'relation_2D':
            self.relation_nn = Relation_metric_2D(self.inchannel,factor = self.factor)

    def forward(self, input, target):
        if self.metric_method == "relation_1D":
            return self.relation_metric_1D(input, target)
        elif self.metric_method =='relation_2D':
            return self.relation_metric_2D(input, target)
        else:
            return pair_metric(input, target, self.metric_method)

    def relation_metric_1D(self, inputs, support):
        """
        :param inputs: the query data features: shape = [batch, feature_dim,length]
        :param support: the support data feature: shape = [num_class, feature_dim,length]
        :return: by using the relation_nn to metric the similarities between query and support: shape = [batch, num_class]
        """
        batch,dim,length = inputs.size()
        num_class = support.size(0)
        input = inputs.unsqueeze(1).expand(batch,num_class,dim,length)
        supports = support.unsqueeze(0).expand(batch,num_class,dim,length)
        final_input = torch.cat((supports,input),dim=2).view(-1,dim*2,length)
        x = self.relation_nn(final_input)
        out = x.view(-1,num_class)
        return out

    def relation_metric_2D(self,input,support):
        """
        :param input: the query data features: shape = [batch, dim, height, width]
        :param support: the support data feature: shape = [num_class, dim, height, width]
        :return: by using the relation_nn to metric the similarities between query and support: shape = [batch, num_class]
        """
        batch_size,dim,height,width = input.size()
        num_class = support.size(0)
        input_ex = input.unsqueeze(1).repeat(
            1, num_class, 1, 1, 1
        )  # shape : [batch, num_class, dim, height, width]
        support_ex = support.unsqueeze(0).repeat(
            batch_size, 1, 1, 1, 1
        )  # shape : [batch, num_class, dim, height, width]
        final_input = torch.cat((support_ex,input_ex),dim=2).view(-1,dim*2,height,width)
        x = self.relation_nn(final_input)
        out = x.view(-1, num_class)
        return out


# implementing the basic measure methods: Euclidean & cosineSimilarity
def pair_metric(inputs,support,metric_method):
    """
    :param inputs: the training or testing output of size [batch,dim]
    :param support: the support set data of size [num_class,dim]
    :param metric_method: the method to metric the similarity of tew data: cosine, Euclidean distance, ...
    :return: return the predict label of inputs based on the similarity between the inputs and support.
    """
    if metric_method == "cosine":
        similarities = cosine_similarity(inputs,support)   # similarity of size [batch,num_class*k_shot]
    else:
        similarities = euclidean_similarity(inputs,support)

    return similarities


def euclidean_similarity(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)

    # x,y may have more than two dimensions
    x = x.view(n, -1)
    y = y.view(m, -1)

    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return -torch.pow(x - y, 2).sum(2)


def cosine_similarity(x, y):
    '''
    Compute cosine similarity between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)

    # x,y may have more than two dimensions
    x = x.view(n, -1)
    y = y.view(m, -1)

    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    cosinesimilarity = nn.CosineSimilarity(dim=2, eps=1e-6)

    return cosinesimilarity(x, y)


def make_divisible(x, divisible_by=8):
    import numpy as np

    return int(np.ceil(x * 1.0 / divisible_by) * divisible_by)


def conv_1x1_bn(
        inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU
):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup, momentum=1, affine=True),
        nlin_layer(inplace=True),
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


# implemting the relation measure net work for feature whose shape is [dim,length]
class Relation_metric_1D(nn.Module):
    def __init__(self,inchannel):
        super(Relation_metric_1D, self).__init__()
        self.relation_nn = nn.Sequential(
            nn.Conv1d(inchannel, inchannel*3, kernel_size=3, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm1d(inchannel*3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(inchannel*3, inchannel*2, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.ReLU(inplace=True),

            nn.Flatten(),
            nn.Dropout(p=0.8),
            nn.Linear(inchannel*2,1),
            nn.Sigmoid(),
        )

    def forward(self,x):
        return self.relation_nn(x)


# implemting the relation measure net work for feature whose shape is [dim,height,width]
class Relation_metric_2D(nn.Module):
    def __init__(self, inchannel, factor=0.25, ):
        super(Relation_metric_2D, self).__init__()
        last_conv = make_divisible(576 * factor)
        self.features = []
        self.features.append(
            conv_1x1_bn(inchannel, last_conv, nlin_layer=Hswish)
        )
        # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
        self.features.append(nn.AdaptiveAvgPool2d(1))
        self.features.append(
            nn.Conv2d(last_conv, make_divisible(128 * factor), 1, 1, 0)
        )
        self.features.append(Hswish(inplace=True))
        self.features.extend(
            [
                nn.Flatten(),
                nn.Dropout(p=0.8),
                nn.Linear(make_divisible(128 * factor), 1),
                nn.Sigmoid(),
            ]
        )

        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        x = self.features(x)
        return x


if __name__ == '__main__':
    # input1 = torch.Tensor([[1,1,1,1,1,1,1],[2,2,2,2,2,2,2]])
    # input2 = torch.Tensor([[3,3,3,3,3,3,3],[4,4,4,4,4,4,4]])

    # a = Pair_metric(metric_method="relation_nn",num_target=2)
    # print(a(input1,input2))

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    # model = Relation_metric_2D(192).to(device)
    # summary(model, (192, 8, 50))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    model = Relation_metric_1D(54).to(device)
    summary(model, (54, 50))

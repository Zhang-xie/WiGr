from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable


def top1_accuracy(inputs,target,batch_size):
    prediction = inputs.data.max(1)[1]
    correct = prediction.eq(target.data.squeeze().data.long()).sum()
    return float(correct) / batch_size


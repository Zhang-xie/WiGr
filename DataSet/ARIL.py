from __future__ import absolute_import, print_function
"""
ARILdata
"""
import torch
import torch.utils.data as data
from pathlib import Path
import os
import sys
sys.path.append('/home/yk/zx/Domain adaptation CSI V4/DataSet/')
import torch.nn as nn
import numpy as np
import itertools,functools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import TensorDataset, DataLoader
from CustomDataset import CustomIterDataset

def default_loader(root):   #root is the data storage path  eg. path = D:/CSI_Data/signfi_matlab2numpy/
    train_label_activity_path = root/"datatrain_activity_label.npy"
    train_amp_path = root/ "datatrain_data.npy"
    train_label_loc_path = root/"datatrain_location_label.npy"
    test_label_activity_path = root/"datatest_activity_label.npy"
    test_amp_path = root/"datatest_data.npy"
    test_label_loc_path = root/"datatest_location_label.npy"

    train_label_activity = np.load(train_label_activity_path)
    train_amp = np.load(train_amp_path)
    train_label_loc = np.load(train_label_loc_path)
    test_label_activity = np.load(test_label_activity_path)
    test_amp = np.load(test_amp_path)
    test_label_loc = np.load(test_label_loc_path)
    return train_label_activity,train_amp,train_label_loc,test_label_activity,test_amp,test_label_loc


def loader_all(root):   #root is the data storage path  eg. path = D:/CSI_Data/signfi_matlab2numpy/
    train_label_activity_path = root/"datatrain_activity_label.npy"
    train_amp_path = root/"datatrain_data.npy"
    train_label_loc_path = root/"datatrain_location_label.npy"
    test_label_activity_path = root/"datatest_activity_label.npy"
    test_amp_path = root / "datatest_data.npy"
    test_label_loc_path = root/ "datatest_location_label.npy"

    train_label_activity = np.load(train_label_activity_path)
    train_amp = np.load(train_amp_path)
    train_label_loc = np.load(train_label_loc_path)
    test_label_activity = np.load(test_label_activity_path)
    test_amp = np.load(test_amp_path)
    test_label_loc = np.load(test_label_loc_path)

    all_label_activity = np.concatenate((train_label_activity,test_label_activity))
    all_amp = np.concatenate((train_amp,test_amp))
    all_label_location = np.concatenate((train_label_loc, test_label_loc))
    return all_label_activity,all_label_location,all_amp



class ARIL(CustomIterDataset):
    def __init__(self, root, roomid=None,userid=None,location=None,orientation=None,receiverid=None,sampleid=None,
                 data_shape=None,chunk_size=50,num_shot=1,batch_size=50,mode=None,loader=loader_all,trainmode=None,trainsize=0.8):
        """
        :param root: dataset storage path
        :param roomid: useless
        :param userid: useless
        :param location: choosing the location: {0,1,2,...,15}
        :param orientation: useless
        :param receiverid:useless
        :param sampleid:useless
        :param data_shape: if datashape = '1D' we using the subcarry x time data
        :param chunk_size:setting the length of every chunk on the time dimension.
        :param num_shot:the number of samples of each gesture(class) in the support set
        :param batch_size: the number of samples per class. hence that, the real_batch_size = batch_size * num_class
        :param mode: useless
        :param loader:
        """

        super().__init__(trainmode,trainsize)



        self.root = root
        self.load = loader
        self.data_shape = data_shape
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.num_shot = num_shot
        self.num_class = 6
        self.batch_idx = 0

        all_label_activity,  all_label_location, self.all_amp = loader_all(root)
        self.all_label_activity = np.squeeze(all_label_activity)
        self.all_label_location = np.squeeze(all_label_location)
        self.total_samples = len(self.all_label_activity)
        self.loc_select = np.ones(self.total_samples, dtype=np.bool)
        index_temp = np.arange(self.total_samples)

        if location is not None:
            self.loc_select = functools.reduce(np.logical_or,[*[self.all_label_location == j for j in location]])

        self.index = index_temp[self.loc_select]
        np.random.shuffle(self.index)

        choosed_label = self.all_label_activity[self.index]
        num_sample_per_class = []
        self.sample_index_per_class = []
        for i in range(0,self.num_class):
            temp = self.index[np.where(choosed_label == i)]
            num_sample_per_class.append(len(temp))
            self.sample_index_per_class.append(temp)
        self.min_num_sample_class = min(num_sample_per_class)  # find the minimal number of samples of all classes
        self.num_batch = self.min_num_sample_class // self.batch_size

    def get_item(self, index):
        sample_index = index
        activity_label_index = self.all_label_activity[sample_index]

        ges_label = torch.tensor(activity_label_index).type(torch.LongTensor)
        data_index = self.all_amp[sample_index]  # shape [52,196]

        if self.data_shape == '1D':
            sample = torch.from_numpy(data_index).type(torch.FloatTensor)  # shape [52,196]
        else:
            sample = torch.from_numpy(data_index).type(torch.FloatTensor)  # shape [52,196]

        return sample,ges_label,

    def metric_data(self):
        # sampling a batch data and split to supportset and training/testing set
        query_data=[]
        query_ges_label = []

        supports_data = []
        supports_ges_label = []
        for i in range(self.num_class):
            temp = self.sample_index_per_class[i][self.batch_idx*self.batch_size:(self.batch_idx+1)*self.batch_size]
            for j in range(0,self.batch_size-self.num_shot):
                sample, ges_label = self.get_item(temp[j])
                query_data.append(sample)
                query_ges_label.append(ges_label)
            for k in range(self.batch_size-self.num_shot,self.batch_size):
                sample, ges_label = self.get_item(temp[k])
                supports_data.append(sample)
                supports_ges_label.append(ges_label)

        self.batch_idx += 1
        return (query_data,query_ges_label),(supports_data,supports_ges_label)

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_idx > self.num_batch-1:
            self.batch_idx = 0
            raise StopIteration
        return self.metric_data()

    def __len__(self):
        return self.num_batch


if __name__ == "__main__":
    root = Path("../../Datasets/ARIL")
    a = ARIL(root=root,location=[13],chunk_size=30,num_shot=1,batch_size=5,data_shape='1D')

    print(a.num_batch)
    print(len(a.index))
    print(a.min_num_sample_class)

    tr_loader = DataLoader(dataset=a,collate_fn=lambda x:x,)
    for i,x in enumerate(tr_loader):
        x = x[0]
        print(len(x))  # query set , support set
        print(len(x[0]))  # query set : 2   (list.data,list.ges_label)
        print(len(x[1]))  # support set  : 2   (list.data,list.ges_label)
        print(len(x[1][0]))  # 12
        print(len(x[0][0]))  # 228
        print(x[0][0][0].shape)  # torch.Size([7, 1, 30, 52])
        print(x[0][1][0].shape)  # torch.Size([])
        print(x[0][0][0])
        break





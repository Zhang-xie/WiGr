import zarr
import torch.utils.data
import zipfile
from pathlib import Path
import numpy as np
import io
import torch.utils.data as data
import torch
import os
import itertools,functools
from torch.utils.data import TensorDataset, DataLoader
from CustomDataset import CustomIterDataset


def split_array_bychunk(array, chunksize, include_residual=True):
    len_ = len(array) // chunksize * chunksize
    array, array_residual = array[:len_], array[len_:]
    # array = np.split(array, len_ // chunksize)
    array = [
        array[i * chunksize: (i + 1) * chunksize]
        for i in range(len(array) // chunksize)
    ]
    if include_residual:
        if len(array_residual) == 0:
            return array
        else:
            return array + [
                array_residual,
            ]
    else:
        if len(array_residual) == 0:
            return array, None
        else:
            return array, array_residual


class CSI_301(CustomIterDataset):
    def __init__(self, root, roomid=None,userid=None,location=None,orientation=None,receiverid=None,sampleid=None,
                 data_shape=None,chunk_size=None,num_shot=1,batch_size=50,mode=None,trainmode=None,trainsize=0.8):
        """
        :param root: the path of the datasets
        :param roomid: choosing the data from specific room, i.e. roomid = [1,2] for choosing the data from room1 & room2
        :param userid: choosing the data from specific user, i.e. userid = [1,2] for choosing the data from user1 & user2
        :param location: choosing the data from specific location, i.e. location = [1,2] for choosing the data from location1 & location2
        :param orientation: useless
        :param receiverid: useless
        :param sampleid: useless
        :param data_shape: if data_shape='split': we using repeat x chunck x 1 x subcarry data; else: '1D' we using the subcarry x time data; else '2D' we using the links X subcarry X time
        :param chunk_size: setting the length of every chunk on the time dimension.
        :param num_shot: the number of samples of each gesture(class) in the support set
        :param batch_size: the number of samples of each gesture(class) in both query set and support set.
        :param mode: if mode == "phase", we only use phase value. if mode == "amplitude", we only use amplitude value. otherwise, we use both.
        """


        super().__init__(trainmode,trainsize)
        
        self.root = root
        self.data_shape = data_shape
        self.chunk_size = chunk_size
        self.num_shot = num_shot
        self.batch_size = batch_size
        self.batch_idx = 0
        self.num_class = 6
        self.mode = mode

        self.group = zarr.open_group(root.as_posix(), mode="r")
        # if self.mode == "amplitude":
        #     self.amp = self.group.csi_data_amp[:]
        self.gesture = self.group.csi_label_act[:]  # 动作 6个 0~5
        self.room_label = self.group.csi_label_env[:]  # 房间 0,1
        self.location = self.group.csi_label_loc[:]  # 站位 0,1  0,1,2
        self.userid = self.group.csi_label_user[:]  # 人 0,1,2,3,4

        self.total_samples = len( self.gesture)
        self.select = np.ones(self.total_samples, dtype=np.bool)
        self.room_select = np.ones(self.total_samples, dtype=np.bool)
        self.user_select = np.ones(self.total_samples, dtype=np.bool)
        self.loc_select = np.ones(self.total_samples, dtype=np.bool)

        index_temp = np.arange(self.total_samples)

        if roomid is not None:
            self.room_select = functools.reduce(np.logical_or,[*[self.room_label == j for j in roomid]])
            self.select = np.logical_and(self.select,self.room_select)
        if userid is not None:
            self.user_select = functools.reduce(np.logical_or,[ *[self.userid == j for j in userid]])
            self.select = np.logical_and(self.select, self.user_select)
        if location is not None:
            self.loc_select = functools.reduce(np.logical_or,[*[self.location == j for j in location]])
            self.select = np.logical_and(self.select, self.loc_select)

        self.index = index_temp[self.select]  # the data for a specified task
        np.random.shuffle(self.index)

        choosed_label = self.gesture[self.index]
        num_sample_per_class = []
        self.sample_index_per_class = []
        for i in range(0, self.num_class):
            temp = self.index[np.where(choosed_label == i)]
            num_sample_per_class.append(len(temp))
            self.sample_index_per_class.append(temp)
        self.min_num_sample_class = min(num_sample_per_class)  # find the minimal number of samples of all classes
        self.num_batch = self.min_num_sample_class // self.batch_size

    def get_item(self, sample_index):
        if self.mode is not None:
            if self.mode == 'phase':
                pha_sample = self.group.csi_data_pha[sample_index]
                sample = pha_sample.astype(np.float32)  # shape [1800,3,114]
            elif self.mode == 'amplitude':
                amp_sample = self.group.csi_data_amp[sample_index]
                # amp_sample = self.amp[sample_index]
                sample = amp_sample.astype(np.float32)  # shape [1800,3,114]
            else:
                amp_sample = self.group.csi_data_amp[sample_index]
                pha_sample = self.group.csi_data_pha[sample_index]
                amp_sample = amp_sample.astype(np.float32)  # shape [1800,3,114]
                pha_sample = pha_sample.astype(np.float32)  # shape [1800,3,114]
                sample = np.concatenate((amp_sample, pha_sample), axis=2)   # shape [1800,3,228]
        else:
            amp_sample = self.group.csi_data_amp[sample_index]
            pha_sample = self.group.csi_data_pha[sample_index]
            amp_sample = amp_sample.astype(np.float32)  # shape [1800,3,114]
            pha_sample = pha_sample.astype(np.float32)  # shape [1800,3,114]
            sample = np.concatenate((amp_sample, pha_sample), axis=2)  # shape [1800,3,228]

        ges_label = self.gesture[sample_index]    # {0,1, 2, 3, 4, 5}
        ges_label = torch.tensor(ges_label).type(torch.LongTensor)

        if self.data_shape == 'split':
            samp, samp_res = split_array_bychunk(sample, self.chunk_size,
                                                 include_residual=False)  # shape list{[chunk,3,114],repeat}
            samp += [sample[-self.chunk_size:], ]
            sample = torch.Tensor(np.array(samp)).type(torch.FloatTensor)
            sample = sample.permute(0, 2, 1, 3).type(torch.FloatTensor)  # shape [repeat,3,chunk,114 or 2x114]
        elif self.data_shape == '1D':
            sample = torch.from_numpy(sample).type(torch.FloatTensor)
            if self.mode == "phase" or self.mode == "amplitude":
                sample = sample.permute(1, 2, 0).reshape(342, -1)  # shape [3X114, 1800]
            else:
                sample = sample.permute(1, 2, 0).reshape(684, -1)  # shape [3X2x114, 1800]
        elif self.data_shape == '2D':
            sample = torch.from_numpy(sample).type(torch.FloatTensor)
            sample = sample.permute(1, 2, 0)
        else:
            sample = torch.from_numpy(sample).type(torch.FloatTensor)
            if self.mode == "phase" or self.mode == "amplitude":
                sample = sample.permute(1, 2, 0).reshape(342, -1)  # shape [3X114, 1800]
            else:
                sample = sample.permute(1, 2, 0).reshape(684, -1)  # shape [3X2x114, 1800]

        return sample,ges_label,

    def metric_data(self):
        # sampling a batch data and split to supportset and training/testing set
        datas_data=[]
        datas_ges_label = []

        supports_data = []
        supports_ges_label = []
        for i in range(self.num_class):
            temp = self.sample_index_per_class[i][self.batch_idx*self.batch_size:(self.batch_idx+1)*self.batch_size]
            for j in range(0,self.batch_size-self.num_shot):
                sample, ges_label = self.get_item(temp[j])
                datas_data.append(sample)
                datas_ges_label.append(ges_label)
            for k in range(self.batch_size-self.num_shot,self.batch_size):
                sample, ges_label = self.get_item(temp[k])
                supports_data.append(sample)
                supports_ges_label.append(ges_label)

        self.batch_idx += 1
        return (datas_data,datas_ges_label),(supports_data,supports_ges_label)

    def get_choose_label(self,id):
        if id == "user":
            return self.userid[self.index]
        if id == "room":
            return self.room_label[self.index]
        if id == "location":
            return self.location[self.index]
        if id == "gesture":
            return self.gesture[self.index]

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_idx > self.num_batch-1 :
            self.batch_idx = 0
            raise StopIteration
        return self.metric_data()

    def __len__(self):
        return self.num_batch


if __name__ == "__main__":
    root = Path("F:/CSI_301")
    a = CSI_301(root=root,roomid=[0],userid=[0],location=[0],data_shape='split',chunk_size=30,num_shot=1,batch_size=5,mode="amplitude")

    print(a.num_batch)
    print(len(a.index))
    print(a.min_num_sample_class)

    tr_loader = DataLoader(dataset=a,collate_fn=lambda x:x)

    print("room:", np.unique(a.get_choose_label("room")))
    print("gesture:", np.unique(a.get_choose_label("gesture")))
    print("user:", np.unique(a.get_choose_label("user")))
    print("location:", np.unique(a.get_choose_label("location")))

    for i,x in enumerate(tr_loader):
        x = x[0]
        print(len(x))
        print(len(x[0]))  # data for train/test  : 2   (amp,ges_label)
        print(len(x[1]))  # support data  : 2   (amp,ges_label)
        print(len(x[1][0]))  # 6
        print(len(x[0][0]))  # 24
        print(x[0][0][0].shape)  # torch.Size([61, 3, 30, 228])
        print(x[0][1][0].shape)  # torch.Size([])
        break



import torch.utils.data
import numpy as np

from numpy.random import RandomState

class CustomIterDataset(torch.utils.data.IterableDataset):
    rs_seed=np.random.randint(0,10000)
    def __init__(self,trainmode,trainsize):
        super().__init__()
        self.rt=RandomState(self.rs_seed)
        self.trainmode=trainmode
        self.trainsize=trainsize
    @property
    def index(self):
        return self.__index

    @index.setter
    def index(self,x):
        self.__index=np.copy(x)
        print(len(x))
        if self.trainmode is not None:
            self.rt.shuffle(self.__index)
            self.__index=self.__index[:int(len(self.__index)*self.trainsize)] if self.trainmode else self.__index[int(len(self.__index)*self.trainsize):]
            print(len(self.index))
        # self.rt.shuffle(self.__index)
        # self.__index=self.__index[:int(len(self.__index)*self.trainsize)] if self.trainmode else self.__index[int(len(self.__index)*self.trainsize):]
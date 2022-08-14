import os
import numpy as np
from sklearn import preprocessing
import netCDF4 as nc
from netCDF4 import Dataset
from net2_utils import *
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch
transform=transforms.Compose([
    transforms.ToTensor()
])

#获取（720,1440）数据
class Unet_Dataset():
    def __init__(self,path):
        self.path=path
        self.name1=os.listdir(os.path.join(path,'ASRMEdata'))#标签，预测
        self.name2 = os.listdir(os.path.join(path, 'Modeldata'))
        self.name3 = os.listdir(os.path.join(path, 'obs_data'))
#路径拼接，listdir是所有文件
    def __len__(self):
        return len(self.name1)

    def __getitem__(self, index):
        name1=self.name1[index]#xxx.nc#标签数据
        name2=self.name2[index]
        name3=self.name3[index]
        asrmedata_path=os.path.join(self.path,'ASRMEdata',name1)#标签拼接地址 预测1998.1-2014.12，204
        Modeldata_path=os.path.join(self.path,'Modeldata',name2)#源数据地址   观测1997.12-2014.11，204
        obsdata_path = os.path.join(self.path, 'obs_data', name3)  # 源数据地址 观测1998.1-2014.12，204
        Modeldata=Dataset(Modeldata_path)
        mod_data = Normalization(Modeldata['tos'][:])
        ASRMEdata=Dataset(asrmedata_path)
        SICdata = Normalization(ASRMEdata['tos'][:])
        obsdata = Dataset(obsdata_path)
        obs_data = Normalization(obsdata['tos'][:])
        return transform(mod_data),transform(SICdata),transform(obs_data)


def Normalization(X):
    min_max_scaler = preprocessing.MinMaxScaler()
    X_minMax = min_max_scaler.fit_transform(X)
    return X_minMax

def load_data(train_data_path, batch_size):
    train_set = Unet_Dataset(train_data_path)
    # 验证集分配
    train_loader,val_loader = val_set_alloc(train_set,batch_size)
    return train_loader, val_loader

#验证集分配
def val_set_alloc(dataset,batch_size):
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_set, batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)

    return train_loader,val_loader
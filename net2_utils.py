import numpy as np
import os
import netCDF4 as nc
import torch


def keep_data_size(mod_data,obs_data):
    a = torch.cat((mod_data, obs_data), 1)
    b = a.to(torch.float32)
    return b

def write_nc(out_folder,newName,output,SICdata):
    # 看训练效果的一个变化
    a = np.array(output.cpu().detach().numpy())  # tensor转换成array
    SICdata = np.array(SICdata.cpu().detach().numpy())

    pathOut = os.path.join(out_folder, newName)

    f_w = nc.Dataset(pathOut, 'w', format='NETCDF4')

    # 确定基础变量的维度信息。相对与坐标系的各个轴(x,y,z)
    f_w.createDimension('Y', 720)
    f_w.createDimension('X', 1440)

    ##创建变量。参数依次为：‘变量名称’，‘数据类型’，‘基础维度信息’
    f_w.createVariable('Y', np.float64, ('Y'))
    f_w.createVariable('X', np.float64, ('X'))

    # 写入变量Y的数据。
    Y = np.arange(0, 720)
    f_w.variables['Y'][:] = Y
    # 写入变量X的数据
    X = np.arange(0, 1440)
    f_w.variables['X'][:] = X

    # 新创建一个多维度变量，并写入数据
    f_w.createVariable('SIC', np.float64, ('Y', 'X'))
    f_w.createVariable('tos', np.float64, ('Y', 'X'))

    f_w.variables['SIC'][:] = a
    f_w.variables['tos'][:] = SICdata
    f_w.close()



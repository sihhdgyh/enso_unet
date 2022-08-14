import math
import torch
import matplotlib as mpl
from net2_utils import *
from Unet_data import *
from net2 import *

mpl.use('Agg')
import matplotlib.pyplot as plt

# 数据集
train_data_path = r'F:/Unet2/data/'
train_loader, val_loader= load_data(train_data_path,batch_size=1)

# 定义超参数
learning_rate = 0.001
num_epoch = 50
batch_size = 1
best_loss=1000

# 定义网络
net = UNet(2,1)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
net.to(device)
# 交叉熵损失函数
loss_fun = torch.nn.MSELoss()
# SGD梯度下降方法
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# 训练和评估
losslist = []
val_losslist = []

for epoch in range(num_epoch):
    running_loss = 0.0
    net.train()
    for i, (mod_data, SICdata, obs_data) in enumerate(train_loader):
        mod_data, obs_data, SICdata = mod_data.to(device), obs_data.to(device), SICdata.to(device)
        optimizer.zero_grad()
        input = keep_data_size(mod_data, obs_data)
        output = net(input)


        loss = loss_fun(output.float(), SICdata.float())
        RMSE = torch.sqrt(loss_fun(output.float(), SICdata.float()))

        RMSE.backward()
        optimizer.step()
        running_loss += loss.item()
        j = i
        # 每隔五次拿一次权重
        if j % 5 == 0:
            print(f'{epoch}-{j}-train_loss===>>{RMSE.item()}')
        i += 1
    train_RMSE_loss = math.sqrt(running_loss / i)
    print('%d epoch:%f' % (epoch + 1, train_RMSE_loss))
    losslist.append(train_RMSE_loss)
    f1 = open("train_loss.txt", 'a')
    f1.write(str(train_RMSE_loss))
    f1.write("\n")
    f1.close()

    sum_loss = 0
    #accurate = 0
    net.eval()
    with torch.no_grad():
        for i, (mod_data, SICdata, obs_data) in enumerate(train_loader):
            mod_data, obs_data, SICdata = mod_data.to(device), obs_data.to(device), SICdata.to(device)
            input = keep_data_size(mod_data, obs_data)
            output = net(input)
            loss = loss_fun(output.float(), SICdata.float())
            RMSE = torch.sqrt(loss_fun(output.float(), SICdata.float()))
            sum_loss += loss.item()
            j = i
            i += 1

        val_RMSE_loss = math.sqrt(sum_loss / i)
        val_losslist.append(val_RMSE_loss)
        if(val_RMSE_loss < best_loss):
            torch.save(net, './ConvLstm_day1.pth'.format(epoch + 1))
            print("第{}轮模型训练数据已保存".format(epoch + 1))
            best_loss=val_RMSE_loss

        f2 = open("val_loss.txt", 'a')
        f2.write(str(val_RMSE_loss))
        f2.write("\n")
        f2.close()

#writer.close()

# 画训练误差图
plt.plot(losslist[0:], color='blue', label='train_loss')
plt.plot(val_losslist[0:], color='red', label='val_loss')
plt.xlabel('epoch')
plt.ylabel('RMSE_loss')
plt.legend()
plt.savefig("loss.png")




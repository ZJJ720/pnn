import torch
import torch.nn as nn
from DNN_tools import *
import random
from torch.utils.data.dataloader import DataLoader
import numpy as np
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import *
import copy
from physical_model import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(27)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_length  = 0.7
batch_size    = 10
num_epochs    = 100
step_size     = 5 # 学习率变化周期
gamma         = 0.5 # 学习率更新值
learning_rate = 0.001 # 学习率

# 电池参数
C_Hp0         = 3850
C_Hn0         = 3030
C_H2Op0       = 44600
S0            = 1000
k_p0          = 0.0001
k_n0          = 0.00005
theta_e0      = 500

# 创建 dataset
dataset_1 = create_dataset(1)
dataset_2 = create_dataset(2)
dataset_3 = create_dataset(3)
dataset_4 = create_dataset(4)

# 将数据的前 train_length 的数据作为训练集 train_set
train_size_1 = int(dataset_1.shape[0] * train_length)
train_size_2 = int(dataset_2.shape[0] * train_length)
train_size_3 = int(dataset_3.shape[0] * train_length)
train_size_4 = int(dataset_4.shape[0] * train_length)
test_size = []
test_size.append(dataset_1.shape[0] * (1 - train_length))
test_size.append(dataset_2.shape[0] * (1 - train_length))
test_size.append(dataset_3.shape[0] * (1 - train_length))
test_size.append(dataset_4.shape[0] * (1 - train_length))

# 分离出训练集和测试集
train_data = []
test_data = []
train_data.append(dataset_1[0:train_size_1])
test_data.append(dataset_1[train_size_1:])
train_data.append(dataset_2[0:train_size_2])
test_data.append(dataset_2[train_size_2:])
train_data.append(dataset_3[0:train_size_3])
test_data.append(dataset_3[train_size_3:])
train_data.append(dataset_4[0:train_size_4])
test_data.append(dataset_4[train_size_4:])

train = np.concatenate((train_data[0], train_data[1], train_data[2], train_data[3]), axis=0)
np.random.shuffle(train)

# 数据归一化
scaler, train_scaled = train_scale(train)
dataset = DataPrepare(train_scaled) # 设置 inputs 和 labels
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True)


# 定义全连接神经网络类

class DNN(nn.Module):
    def __init__(self, input_dim=5, output_dim=4, hidden_layers=[128]):
        super(DNN, self).__init__()

        # 定义隐藏层
        self.hidden_layers = nn.ModuleList()
        layer_sizes = [input_dim] + hidden_layers
        for i in range(len(hidden_layers)):
            self.hidden_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.hidden_layers.append(nn.ReLU())  # 使用ReLU作为激活函数

        # 定义输出层
        self.output_layer = nn.Linear(hidden_layers[-1], output_dim)

    def forward(self, x):
        for linear in self.hidden_layers:
            x = linear(x)
        output = self.output_layer(x)
        return output


# 自定义损失函数
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, inputs, preds, labels):
        # 自定义损失逻辑
        S = preds[:, 0]
        k_p = preds[:, 1]
        k_n = preds[:, 2]
        theta_e = preds[:, 3]
        S = S0 * torch.exp(S).cpu()
        k_p = k_p0 * torch.exp(k_p).cpu()
        k_n = k_n0 * torch.exp(k_n).cpu()
        theta_e = theta_e0 * torch.exp(theta_e).cpu()
        T, I, SoC, Q, C_V0, U = train_invert_scale(scaler, inputs.view(-1, 5).cpu().numpy(), labels.view(-1, 1).cpu().detach().numpy())
        C_2, C_3, C_4, C_5, C_Hn, C_Hp, C_H2Op = get_con(SoC, C_V0, C_Hp0, C_Hn0, C_H2Op0)
        e_con = E_con(T, I, Q, C_2, C_3, C_4, C_5)
        e_act = E_act(T, I, S, k_p, k_n, C_2, C_3, C_4, C_5)
        e_ohm = E_ohm(theta_e, T, I)
        e_ocv = E_ocv(T, C_2, C_3, C_4, C_5, C_Hp, C_Hn, C_H2Op)
        e_cell = e_con + e_act + e_ohm + e_ocv
        loss = torch.mean((e_cell - U) ** 2)
        return loss



# 创建模型实例
model = DNN().to(device)


# 定义损失函数和优化器
loss_function = CustomLoss()
loss_function = loss_function.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam 优化器
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

# 训练模型
min_epochs = 10
best_model = None
min_loss = 1

for epoch in tqdm(range(num_epochs)):
    train_loss = []
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.reshape(batch_size, 5)

       # 梯度清零
        optimizer.zero_grad()
        inputs = inputs.to(torch.float32)
        labels = labels.to(torch.float32)

        # 前向传播
        preds = model(inputs)

        # 计算损失
        loss = loss_function(inputs, preds, labels)
        train_loss.append(loss.cpu().item())

        # 更新梯度
        loss.backward()

        # 优化参数
        optimizer.step()  # 更新每个网络的权重

    scheduler.step()

    if epoch > min_epochs and loss < min_loss:
        min_val_loss = loss
        best_model = copy.deepcopy(model)

    print('epoch {} train_loss {:.8f}'.format(epoch, train_loss[50]))
    model.train()

    torch.save(model, f'./result/DNN_model.pth')


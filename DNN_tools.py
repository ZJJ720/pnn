from sklearn.preprocessing import MinMaxScaler
from torch.utils.data.dataset import Dataset
import torch
import pandas as pd
import numpy as np


def create_dataset(excel_index):
    # 读取 excel 数据
    df = pd.read_excel(f'data/test{excel_index}.xlsx') # 假设你的数据文件名为 data.xlsx
    # 前 5 列数据读取为 dataX
    dataX = df.iloc[:, [0,1,2,4,5]].values
    # 第 7 列数据读取为 dataY
    dataY = df.iloc[:, 6].values
    # 将 dataX, dataY 变量转变为 tensor 类型
    dataX = torch.from_numpy(dataX).float()
    dataY = torch.from_numpy(dataY).float().reshape(-1, 1)


    dataset = np.concatenate((dataX, dataY), axis=1)

    return dataset


def train_scale(train):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train_scaled = scaler.transform(train)
    return scaler, train_scaled


def test_scale(test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(test)
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, test_scaled


def invert_scale(scaler: object, X: object, value: object) -> object:
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array) # 去标准化
    inverted = torch.Tensor(inverted)
    return inverted[0, 0],inverted[0, 1],inverted[0, 2],inverted[0, 3],inverted[0, 4]


class DataPrepare(Dataset):

    def __init__(self, train):
        self.len = train.shape[0]
        x_set = train[:, 0:-1]
        x_set = x_set.reshape(x_set.shape[0], 1, 6)
        # 数据类型转为 torch 变量
        self.x_data = torch.from_numpy(x_set)
        self.y_data = torch.from_numpy(train[:, [-1]])


    def __getitem__(self, index):
        # 返回 img, label
        return self.x_data[index], self.y_data[index]
    

    def __len__(self):
        return self.len
    

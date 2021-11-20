import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


####################数据提取与预处理###################

class data_preprocess:
    def __init__(self):
        # 数据集路径
        self.train_path = "dataset/train/"
        self.test = "dataset/test/test.csv"
        self.train_list = os.listdir(self.train_path)

        # 提取第一个csv文件(去除第一个数据集，第一个数据集和测试集相同，避免影响结果)
        print("正在提取数据集" + (self.train_path + self.train_list[1]).__str__() + "......")
        self.train_dataset = pd.read_csv(self.train_path + self.train_list[1], compression="gzip")
        self.train_dataset.fillna(method='bfill', inplace=True)

        train_list = self.train_list[2:]
        # 将其余数据集也一并提取，最终存储在train_dataset中
        for c in train_list:
            print("正在提取数据集" + c.__str__() + "......")
            df = pd.read_csv(self.train_path + c, compression="gzip")
            df.fillna(method='bfill', inplace=True)
            self.train_dataset = pd.concat([self.train_dataset, df], axis=0)

        print(self.train_dataset)

        # 提取出特征值和y值
        # 训练集
        self.train_x = self.train_dataset.drop(['12210'], axis=1)
        self.train_y = self.train_dataset['12210'].values

        # 测试集
        print("正在提取测试集数据......")
        self.test_dataset = pd.read_csv(self.test, compression="gzip")
        self.test_dataset.fillna(method='bfill', inplace=True)
        self.test_x = self.test_dataset.drop(['12210'], axis=1)
        self.test_y = self.test_dataset['12210'].values

        # 标准差标准化（standardScale）使得经过处理的数据符合标准正态分布，即均值为0，标准差为1
        Scaler = StandardScaler()
        self.X_train = Scaler.fit_transform(self.train_x)
        self.X_test = Scaler.fit_transform(self.test_x)

    # 获取训练集所有样本特征
    def get_X_train(self):
        return self.X_train

    # 获取训练集所有y值
    def get_train_y(self):
        return self.train_y

    # 获取测试集所有特征
    def get_X_test(self):
        return self.X_test

    # 获取测试集所有y值
    def get_test_y(self):
        return self.test_y

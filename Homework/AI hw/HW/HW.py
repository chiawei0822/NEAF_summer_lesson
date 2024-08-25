import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from os import walk
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


#Load data
train_x = []
train_x_std = []
train_y = []
folder_name = ['Yes', 'No']
i = 0
for folder in folder_name:
    path = 'Data/'+ str(folder) +'/'
    for root, dirs, files in walk(path):
        for f in files:
            filename = path + f
            print(filename)

            acc = scipy.io.loadmat(filename)  #讀取matlab檔案
            acc = acc['tsDS'][:,1].tolist()[0:7500]  #把第一行(時間)所有數據取前7500個
            train_x.append(acc)
            train_x_std.append(np.std(acc))
            
            
            if folder == 'Yes':    
                train_y.append(1)
                title = 'Original Signal With Chatter #'
                saved_file_name = 'Figure/Yes_'
            
            if folder == 'No':
                train_y.append(0)
                title = 'Original Signal Without Chatter #'
                saved_file_name = 'Figure/No_'
                
            # # plt.clf()
            # plt.figure(figsize=(7,4))
            # plt.plot(acc, 'b-', lw=1)
            # plt.title(title + str(i+1))
            # plt.xlabel('Samples')
            # plt.ylabel('Acceleration')
            # plt.savefig(saved_file_name + str(i+1) + '.png')                
            # # plt.show()
            # i = i + 1

train_x = np.array(train_x_std)  
train_y = np.array(train_y)
print(train_x)

scaler = MinMaxScaler(feature_range=(0,1))  #數值等比例縮放到0到1之間
train_x = scaler.fit_transform(train_x.reshape(-1,1))  #變成直的
print(train_x)

### homework

#1. 自行建立神經網路
#2. 準確度要達到1
#3. 要交叉驗證
#*如果做不出來建議用CNN*

loo = LeaveOneOut()  #交叉驗證
# model = MLPClassifier(max_iter=500, batch_size=1, solver='adam')
# y_pred = cross_val_predict(model, train_x, train_y, cv=loo)
# y_true = train_y

##


class ComplexNN(nn.Module):
    def __init__(self):
        super(ComplexNN, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(0.5)  # 添加Dropout层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x
    

scaler = MinMaxScaler(feature_range=(0, 1))  # 特徵縮放
train_x = scaler.fit_transform(train_x.reshape(-1, 1))
train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y.reshape(-1, 1), dtype=torch.float32)

y_true = []
y_pred = []

for train_idx, test_idx in loo.split(train_x):
    # 訓練和測試分割
    X_train, X_test = train_x[train_idx], train_x[test_idx]
    y_train, y_test = train_y[train_idx], train_y[test_idx]

    # 初始化模型
    model = ComplexNN()
    criterion = nn.BCELoss()  # 二元交叉熵損失函數
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 訓練過程
    model.train()
    for epoch in range(1000):  # 訓練1000次
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    # 測試過程
    model.eval()
    with torch.no_grad():
        output = model(X_test)
        predicted = torch.round(output)  # 將輸出轉換為0或1
        y_pred.append(predicted.item())
        y_true.append(y_test.item())




print('Prediction: \t', y_pred)
print('Ground Truth: \t', y_true)

cf_m = confusion_matrix(y_true, y_pred)
print('\nConfusion Matrix: \n', cf_m)

#       Confusion  Matrix
#
# GT=0 |    tn   |   fp
# GT=1 |    fn   |   tp
#      +------------------
#         Pred=0   Pred=1
#
#  tn,tp的值越大越好

tn, fp, fn, tp = cf_m.ravel()
accuracy = (tn+tp) / (tn+fp+fn+tp)
print('Accuracy: \t', accuracy)
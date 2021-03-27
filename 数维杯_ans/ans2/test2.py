import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset_train = pd.read_csv('600519.SH.csv')
# g = open('000400.SZ.csv', 'r')
# g1 = open('002281.SZ.csv', 'r')
# g2 = open('600519.SH.csv', 'r')
training_set = dataset_train.iloc[:, 1:2].values #从第一行开始读取第二列
# dataset_train.head()
# print(training_set)
# print(min(training_set))
# print(max(training_set))


## Feature Scaling (Normalization)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) #归一化
training_set_scaled = sc.fit_transform(training_set)
# print(training_set_scaled[0:60])

X_train = []
y_train = []

for i in range (60,1174):                                   ## start dari 60 pertama
    X_train.append(training_set_scaled[i-60:i, 0])          ## untuk i=60 (maka diambil 0:60) dari indeks 0 sampai 59, pada kolom '0'
    y_train.append(training_set_scaled[i,0])                ## untuk i=60 maka diambil indeks ke 60 pada kolom '0'
X_train, y_train = np.array(X_train), np.array(y_train)     #3 mengubahnya kedalam np array

# reshapping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

########################################
# Building and Training the RNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))

# secoond LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))                # tidak melakukan input shape, karena sudah dilakukan pada first LSTM
regressor.add(Dropout(0.2))

# third
regressor.add(LSTM(units = 50, return_sequences = True))   # Third LSTM sama seperti second LSTM
regressor.add(Dropout(0.2))

# fouth
regressor.add(LSTM(units = 50))                            # pada last LSTM, return_sequences yang digunakan False(Default), dan unitnya tetep 50 bukan 1 karena ini bukan last neuron dari NN
regressor.add(Dropout(0.2))

# 添加输出层
regressor.add(Dense(units = 1))
# 编译
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# 拟合数据 训练10次
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

###########################################
# 预测并可视化

# menggunakan data test set
dataset_test = pd.read_csv('600519.SH.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values


# dataset_test.head()
# dataset_test.info()
# real_stock_price

dataset_total = pd.concat((dataset_train['open'], dataset_test['open']), axis = 0)     # menggabungkan original dataframe (kolom 'open') pada training dan testing
print(len(dataset_train['open']))
print(len(dataset_test['open']))
print(len(dataset_total))

dataset_total[len(dataset_total) - len(dataset_test) - 60]
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
# 60, 1174
for i in range (60,1174):                                     ## start dari 60 sebelum 3rd january, dan 80(60+20) 20nya yaitu jumlah data pada test set
    X_test.append(inputs[i-60:i, 0])                        ## X_test inputnya pada index ke 60 (3rd january)

X_test = np.array(X_test)

## reshaping dalam bentuk 3D
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)                   #prediction


## inverse hasilnya kedalam bentuk original dari bentuk 3D
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# print(len(predicted_stock_price))

# print(predicted_stock_price)
import csv
gg = csv.writer(open('predict2.csv', 'w'))

for i in predicted_stock_price :
    gg.writerow(i)
    # print(i)

#gg.close()
print('ok')
#######################
# 可视化
plt.plot(real_stock_price, color = 'red', linestyle = '-', label = 'Real Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Price')
plt.title('Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.savefig('第3只')
plt.show()
# 삼성전자와 코스피를 DNN으로 앙상블

import numpy as np
import pandas as pd

samsung = np.load('./samsung/data/samsung.npy')
kospi200 = np.load('./samsung/data/kospi200.npy')

print(samsung)
print(samsung.shape)
print(kospi200)
print(kospi200.shape)

def split_xy5(dataset, time_steps, y_column): # 5 by 5 로 잘라서 다음 걸 예측
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :] # 5 행의 모든 열 정보 가져옴
        tmp_y = dataset[x_end_number:y_end_number, 3] # 다음 행의 종가 정보
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x1, y1 = split_xy5(samsung, 5, 1)
x2, y2 = split_xy5(kospi200, 5, 1)

# print(x.shape)
# print(y.shape)
# print(x[0,:], "\n", y[0])

# 데이터셋 나누기
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1,y1,random_state=1,test_size=0.3, shuffle=False)
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2,y2,random_state=1,test_size=0.3, shuffle=False)

# print(x_train.shape)
# print(x_test.shape)

# 3차원 -> 2차원 // standardscaler를 하기 위해 2차원으로 변환
x1_train = np.reshape(x1_train,
                     (x1_train.shape[0], x1_train.shape[1] * x1_train.shape[2]))
x1_test = np.reshape(x1_test,
                    (x1_test.shape[0], x1_test.shape[1] * x1_test.shape[2]))
x2_train = np.reshape(x2_train,
                     (x2_train.shape[0], x2_train.shape[1] * x2_train.shape[2]))
x2_test = np.reshape(x2_test,
                    (x2_test.shape[0], x2_test.shape[1] * x2_test.shape[2]))
# print(x_train.shape)
# print(x_test.shape)


# 데이터 전처리
# StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x1_train)
x1_train_scaled = scaler.transform(x1_train)
x1_test_scaled = scaler.transform(x1_test)

scaler = StandardScaler()
scaler.fit(x2_train)
x2_train_scaled = scaler.transform(x2_train)
x2_test_scaled = scaler.transform(x2_test)

print(x1_train_scaled.shape)
print(x2_train_scaled.shape)
print(y1_test.shape)
# # 2차원에서 다시 원래로 복귀 
# x1_train_scaled = np.reshape(x1_train_scaled,
#                      (x1_train_scaled.shape[0], 5, 5))
# x1_test_scaled = np.reshape(x1_test_scaled,
#                      (x1_test_scaled.shape[0], 5, 5))   
# x2_train_scaled = np.reshape(x2_train_scaled,
#                      (x2_train_scaled.shape[0], 5, 5))
# x2_test_scaled = np.reshape(x2_test_scaled,
#                      (x2_test_scaled.shape[0], 5, 5))   

# 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM
               
input1 = Input(shape=(25,))
dense1 = Dense(64)(input1)
dense1 = Dense(64)(dense1)
dense1 = Dense(64)(dense1)
output1 = Dense(32)(dense1)

input2 = Input(shape=(25,))
dense2 = Dense(64)(input2)
dense2 = Dense(64)(dense2)
dense2 = Dense(64)(dense2)
output2 = Dense(32)(dense2)

from keras.layers.merge import concatenate
merge = concatenate([output1, output2])
output3 = Dense(1)(merge)

model = Model(inputs=[input1, input2], outputs = output3 )

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)
model.fit([x1_train_scaled, x2_train_scaled], y1_train, validation_split=0.2, epochs=100, batch_size=1, verbose=1,
          callbacks=[early_stopping])

loss, mse = model.evaluate( [x1_test_scaled, x2_test_scaled] , y1_test, batch_size=1)
print('loss : ', loss)
print('mse : ', mse)

y_pred = model.predict( [x1_test_scaled, x2_test_scaled] )

for i in range(5):
    print('종가 : ', y1_test[i], '/ 예측가 : ', y_pred[i])
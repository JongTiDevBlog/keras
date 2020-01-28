#1. 데이터
import numpy as np

x1 = np.array([range(1,101), range(101, 201), range(301, 401)])
x2 = np.array([range(1001,1101), range(1101, 1201), range(1301, 1401)])

y1 = np.array(range(1101,1201))

# print(x.shape)
# print(y.shape)

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)


from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(
    x1, x2, y1, random_state=66, test_size=0.4, shuffle = False)
x1_val, x1_test, x2_val, x2_test, y1_val, y1_test = train_test_split(
    x1_test, x2_test, y1_test, random_state=66, test_size=0.5, shuffle = False)
 


#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
#model = Sequential()

input1 = Input(shape=(3,))
dense1 = Dense(5)(input1)
dense2 = Dense(2)(dense1)
dense3 = Dense(3)(dense2)
output1 = Dense(1)(dense3)

input2 = Input(shape=(3,))
dense21 = Dense(7)(input2)
dense22 = Dense(4)(dense21)
output2 = Dense(5)(dense22)

from keras.layers.merge import concatenate
merge1 = concatenate([output1, output2])

middle1 = Dense(4)(merge1)
middle2 = Dense(7)(middle1)
output = Dense(1)(middle2)

model = Model(inputs = [input1, input2], outputs = output)

# model.add(Dense(5, input_dim = 1))
# model.add(Dense(32, input_shape = (3, )))
# model.add(Dense(24))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(2))

model.summary()   

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])    # mse, mae
model.fit([x1_train, x2_train], y1_train, epochs=100, batch_size=1, validation_data=([x1_val, x2_val], y1_val))

#4. 평가예측
loss, mse = model.evaluate([x1_test, x2_test], y1_test, batch_size=1)
print('mse : ', mse)


x1_prd=np.array([[1401,1402,1403], [1404,1405,1406], [1407,1408,1409]])
x2_prd=np.array([[1501,1502,1503], [1504,1505,1506], [1507,1508,1509]])
x1_prd = np.transpose(x1_prd)
x2_prd = np.transpose(x2_prd)

aaa=model.predict([x1_prd, x2_prd], batch_size=1)
print(aaa)

# bbb = model.predict(x, batch_size=1)
# print(bbb)

#RMSE 구하기
from sklearn.metrics import mean_squared_error

y1_predict = model.predict([x1_test, x2_test], batch_size=1)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y1_test, y1_predict))

#R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y1_test, y1_predict)
print('R2 : ', r2_y_predict)
#1. 데이터
import numpy as np

x = np.array([range(1,101), range(101, 201), range(301, 401)])
y = np.array([range(1,101), range(101, 201)])
y2 = np.array(range(101, 201))

x = np.transpose(x)
y = np.transpose(y)
y2 = np.transpose(y2)

print(x.shape)
print(y.shape)
print(y2.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=66, shuffle = False)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=66, shuffle = False)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# model.add(Dense(5, input_dim = 1))
model.add(Dense(32, input_shape = (3, )))
model.add(Dense(24))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(2))

# model.summary()   

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])    # mse, mae
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

#4. 평가예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('mse : ', mse)


x_prd=np.array([[401,402,403],[404,405,406],[407,408,409]])
x_prd = np.transpose(x_prd)

aaa=model.predict(x_prd, batch_size=1)
print(aaa)

# bbb = model.predict(x, batch_size=1)
# print(bbb)

#RMSE 구하기
from sklearn.metrics import mean_squared_error

y_predict = model.predict(x_test, batch_size=1)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

#R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print('R2 : ', r2_y_predict)
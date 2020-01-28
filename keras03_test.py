#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test= np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])

# print(x.shape)
# print(y.shape)


#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# model.add(Dense(5, input_dim = 1))
model.add(Dense(5, input_shape = (1, )))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

# model.summary()    # summary에 대한 설명  bias(절편)의 갯수까지 포함하여 계산하기 때문에



#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])    # mse, mae
model.fit(x_train, y_train, epochs=100, batch_size=1)                       # batch_size 바꿔보기(1, 10, 없애보기)   # batch_size의 default값은 32이다.

#4. 평가예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print('mae : ', mae)


x_prd=np.array([11, 12, 13])
aaa=model.predict(x_prd, batch_size=1)
print(aaa)

# bbb = model.predict(x, batch_size=1)
# print(bbb)

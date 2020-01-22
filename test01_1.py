#1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# print(x.shape)
# print(y.shape)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential();

model.add(Dense(20, input_dim = 1))
model.add(Dense(10))
model.add(Dense(1))


#3. 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mse'])
model.fit(x, y, epochs=300, batch_size=1)

#4. 평가예측
loss, mse = model.evaluate(x, y, batch_size=1)
print('mse : ', mse)

x_prd = np.array([11,12,13])
aaa = model.predict(x_prd, batch_size=1)
print(aaa)

bbb = model.predict(x, batch_size=1)
print(bbb)
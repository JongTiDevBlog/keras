# 실습
# 22_2를 이용하여 LSTM으로 구현
# loss 출력
# 90, 100, 110 을 예측

from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)

dataset = [10,20,30,40,50,60,70,80,90,100]
n_steps = 3

x, y = split_sequence(dataset, n_steps)

for i in range(len(x)):
    print(x[i], y[i])


x = x.reshape((x.shape[0], x.shape[1], 1))

model = Sequential()
model.add(LSTM(32, activation='relu', input_shape=(3, 1))) # (3,1) 열, 몇 개씩 자르는지 를 나타냄
# dense층
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

model.summary()


model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1, verbose=2)
loss = model.evaluate(x,y,batch_size=1)
print(loss)
 
x_input = array([90,100,110]) # (3,)
x_input = x_input.reshape(1,3,1)
 
yhat = model.predict(x_input)
print(yhat)


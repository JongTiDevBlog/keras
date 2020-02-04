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


# 실습 dnn 모델 구성
model = Sequential()
model.add(Dense(33, activation = 'relu', input_shape=(3,)))
model.add(Dense(5))
model.add(Dense(1))

# LSTM을 DNN으로 구현 가능 
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=100, batch_size=1, verbose=2)
 
x_input = array([90,100,110]) # (3,)
x_input = x_input.reshape(1,3)
 
yhat = model.predict(x_input)
print(yhat)

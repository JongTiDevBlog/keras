from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 데이터
x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]])
y = array([4, 5, 6, 7, 8])

print(x.shape) # (5,3)
print(y.shape) # (5, )

x = x.reshape((x.shape[0], x.shape[1], 1))
'''
RNN은 은닉층의 결과가 다시 같은 은닉층의 입력으로 들어가도록 연결
FFNets(Feed-Forward Neural Networks)는 현재 주어진 데이터로만 판단
RNN은 지금 들어온 입력데이터와 과거에 입력 받았던 데이터를 동시에 고려
RNN은 장기의존성(Long-Ternm-Denpendency) 문제가 있기 때문에
LSTM(Long Short-Term Memory)는 이의 해법 중 하나
'''

'''
현재 x shape는 (5,3)이지만 input에는 (none,3,1)이 들어가야 함
x를 reshape 한다
'''
# 모델 구성
model = Sequential()
model.add(LSTM(32, activation='relu', input_shape=(3, 1))) # (3,1) 열, 몇 개씩 자르는지 를 나타냄
# 10은 dense층
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

model.summary()
'''
Param: 541   input_dim 3

params = dim(W)+dim(V)+dim(U) = n*n + kn + nm

# n - dimension of hidden layer

# k - dimension of output layer

# m - dimension of input layer
'''

# 실행, 평가 예측
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from keras.callbacks import EarlyStopping, TensorBoard
tb_hist = TensorBoard(log_dir = "./graph",
                      histogram_freq=0,
                      write_graph=True,
                      write_images=True)

early_stopping = EarlyStopping(monitor="loss", patience=10, mode='auto')
model.fit(x, y, epochs=100, batch_size=1, verbose=1,
          callbacks=[early_stopping, tb_hist])
loss, mae = model.evaluate(x,y,batch_size=1)
print(loss, mae)

x_input = array([6,7,8])
#x_input = x_input.reshape((x_input.shape[0], x_input.shape[1], 1))
print(x_input.shape)
#print(x_input.shape[1])
x_input = x_input.reshape(1,3,1)

y_predict = model.predict(x_input)
print(y_predict)
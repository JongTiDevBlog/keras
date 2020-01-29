from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
 
 # 1. 데이터
x1 = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y1 = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x2 = array([[10,20,30], [20,30,40], [30,40,50], [40,50,60], [50,60,70],
            [60,70,80], [70,80,90], [80,90,100], 
            [90,100,110], [100,110,120],
            [2,3,4], [3,4,5], [4,5,6]])
y2 = array([40,50,60,70,80,90,100,110,120,130,5,6,7])


x1 = x1.reshape((x1.shape[0], x1.shape[1], 1))
x2 = x2.reshape((x2.shape[0], x2.shape[1], 1))

# 2. 모델 구성

input1 = Input(shape=(3,1))
model1 = LSTM(10,activation='relu')(input1)
model1 = Dense(5)(model1)
model1 = Dense(2)(model1)
model1 = Dense(3)(model1)
output1 = Dense(3)(model1)

input2 = Input(shape=(3,1))
model2 = LSTM(10,activation='relu')(input2)
model2 = Dense(5)(model2)
model2 = Dense(2)(model2)
model2 = Dense(3)(model2)
output2 = Dense(3)(model2)

# 모델 합치기 concatenate
from keras.layers.merge import concatenate, Add
#merge1 = concatenate([output1, output2])
merge1 = Add()([model1, model2])

middle1 = Dense(4)(merge1)
middle2 = Dense(7)(middle1)
middle3 = Dense(3)(middle2) # 현재 merge된 마지막 레이어

# 분기
output_1 = Dense(30)(middle3) # 1번째 아웃풋 모델
output_1 = Dense(5)(output_1)
output_1 = Dense(1)(output_1)

output_2 = Dense(300)(middle3) # 2번째 아웃풋 모델
output_2 = Dense(5)(output_2)
output_2 = Dense(1)(output_2)

model = Model(inputs = [input1, input2], outputs = [output_1, output_2])
model.summary()   

# 3. 훈련
model.compile(loss='mae', optimizer='adam', metrics=['mae'])
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor="loss", patience=120, mode='auto')
model.fit([x1, x2], [y1, y2], epochs=1000, batch_size=1, verbose=1,
          callbacks=[early_stopping])
# verbose 수에 따라 출력이 다르게 나온다

# 4. 평가 예측
mae = model.evaluate([x1, x2], [y1, y2], batch_size=1)
print(mae)

x1_input = array([[6.5,7.5,8.5],[50,60,70],[70,80,90],
                 [100,110,120], [100, 110, 120]])
x2_input = array([[6.5,7.5,8.5],[50,60,70],[70,80,90],
                 [100,110,120], [100, 110, 120]])

x1_input = x1_input.reshape(5,3,1)
x2_input = x2_input.reshape(5,3,1)

y_predict = model.predict([x1_input, x2_input])
print(y_predict)

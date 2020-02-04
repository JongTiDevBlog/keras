#1. 데이터
import numpy as np

x1 = np.array([range(1,101), range(101,201), range(301,401)])
#x2 = np.array([range(1001,1101), range(1101,1201), range(1301,1401)])

#y1 = np.array(range(1101,1201))

y1 = np.array([range(1,101), range(101,201), range(301,401)])
y2 = np.array([range(1001, 1101), range(1101,1201), range(1301,1401)])
y3 = np.array([range(1,101), range(101,201), range(301,401)])
# print(x.shape)
# print(y.shape)

x1 = np.transpose(x1)
#x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
y3 = np.transpose(y3)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(x1, y1, y2, y3, train_size = 0.6, random_state=66, shuffle=False)
x1_val, x1_test, y1_val, y1_test, y2_val, y2_test, y3_val, y3_test = train_test_split(x1_test, y1_test, y2_test, y3_test, train_size = 0.6, random_state=66, shuffle=False)
# x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=66, test_size=0.4, shuffle = False)
# x1_test, x1_val, y1_test, y1_val = train_test_split(x1_test, y1_test, random_state=66, test_size=0.5, shuffle = False)
 
# x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state=66, test_size=0.4, shuffle = False)
# x2_test, x2_val, y2_test, y2_val = train_test_split(x2_test, y2_test, random_state=66, test_size=0.5, shuffle = False)
 
# y3_train, y3_test = train_test_split(y3, random_state=66, test_size=0.4, shuffle = False)
# y3_test, y3_val = train_test_split(y3_test, random_state=66, test_size=0.5, shuffle = False)

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
#model = Sequential()

input1 = Input(shape=(3,))
dense1 = Dense(5)(input1)
dense2 = Dense(2)(dense1)
dense3 = Dense(3)(dense2)
output1 = Dense(1)(dense3)

# input2 = Input(shape=(3,))
# dense21 = Dense(7)(input2)
# dense22 = Dense(4)(dense21)
# output2 = Dense(5)(dense22)

# 모델 합치기 concatenate
# from keras.layers.merge import concatenate
# merge1 = concatenate([output1, output2])

# middle1 = Dense(4)(merge1)
# middle2 = Dense(7)(middle1)
# middle3 = Dense(3)(middle2) # 현재 merge된 마지막 레이어

# 분기
output_1 = Dense(30)(output1) # 1번째 아웃풋 모델
output_1 = Dense(3)(output_1)

output_2 = Dense(300)(output1) # 2번째 아웃풋 모델
output_2 = Dense(5)(output_2)
output_2 = Dense(3)(output_2)

output_3 = Dense(10)(output1) # 3번째 아웃풋 모델
output_3 = Dense(3)(output_3)

model = Model(inputs = input1, outputs = [output_1, output_2, output_3])

# model.add(Dense(5, input_dim = 1))
# model.add(Dense(32, input_shape = (3, )))
# model.add(Dense(24))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(2))

model.summary()   

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])    # mse, mae
# mse = 평균 제곱 에러, mae = 평균 절대값 에러, 분류 모델에서만 acc를 사용해야 함

from keras.callbacks import EarlyStopping, TensorBoard
tb_hist = TensorBoard(log_dir = "./graph",
                      histogram_freq=0,
                      write_graph=True,
                      write_images=True)

early_stopping = EarlyStopping(monitor="loss", patience=10, mode='auto')
model.fit(x1_train, [y1_train, y2_train, y3_train], epochs=100, batch_size=1, validation_data=(x1_val, [y1_val, y2_val, y3_val]), callbacks=[early_stopping, tb_hist])

#4. 평가예측
mse = model.evaluate(x1_test, [y1_test, y2_test, y3_test], batch_size=1)
# 하나 이상일 경우 loss 와 mse를 리스트 형태로 반환하기 때문에 loss 제거, mse 변수 하나로 리스트 형태로 받음 
print('mse : ', mse)
# 프린트 시, 반환 값 7개인데. 첫 번째는 총 로스, 모델1,2,3의 로스, 모델1,2,3의 mse
# 프린트 시, loss와 mse 값이 같은 이유는 compile 시에 loss = mse 였기 때문이다, loss를 mse로 평가하겠다

x1_prd=np.array([[1401,1402,1403], [1404,1405,1406], [1407,1408,1409]]) # 행은 무시, 열이 중요
x2_prd=np.array([[1501,1502,1503], [1504,1505,1506], [1507,1508,1509]])
x1_prd = np.transpose(x1_prd)
x2_prd = np.transpose(x2_prd)

y1_predict, y2_predict, y3_predict = model.predict(x1_test)

# bbb = model.predict(x, batch_size=1)
# print(bbb)

#RMSE 구하기 , RMSE - 평균 제곱 오차에 루트값을 씌운 것
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

RMSE1=RMSE(y1_test, y1_predict)
RMSE2=RMSE(y2_test, y2_predict)
RMSE3=RMSE(y3_test, y3_predict)
print('RMSE(y1_test) : ', RMSE1)
print('RMSE(y2_test) : ', RMSE2)
print('RMSE(y3_test) : ', RMSE3)
print('AVG(RMSE) : ', (RMSE1+RMSE2+RMSE3)/3)

#R2 구하기, R2 - 회귀 모델에서 예측의 적합도를 0과 1 사이의 값으로 계산한 것. 1은 예측이 완벽한 경우고, 0은 훈련 세트의 출력값인 y_train의 평균으로만 예측하는 모델의 경우
from sklearn.metrics import r2_score
def R2(y_test, y_predict):
    return r2_score(y_test, y_predict)
 
R2_1 = R2(y1_test, y1_predict)
R2_2 = R2(y2_test, y2_predict)
R2_3 = R2(y3_test, y3_predict)
print('R2(y1_test) : ', R2_1)
print('R2(y2_test) : ', R2_2)
print('R2(y3_test) : ', R2_3)
print('AVG(R2) : ', (R2_1+R2_2+R2_3)/3)

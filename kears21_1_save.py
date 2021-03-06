#1. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

model = Sequential()

# input1 = Input(shape=(3,))
# dense1 = Dense(5)(input1)
# dense2 = Dense(2)(dense1)
# dense3 = Dense(3)(dense2)
# output1 = Dense(1)(dense3)

model.add(Dense(5, input_shape = (3, )))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

#model.summary()   

model.save("./save/savetest01.h5")
print("저장 완료")

# #3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['mse'])    # mse, mae

# from keras.callbacks import EarlyStopping, TensorBoard
# tb_hist = TensorBoard(log_dir = "./graph",
#                       histogram_freq=0,
#                       write_graph=True,
#                       write_images=True)

# early_stopping = EarlyStopping(monitor="loss", patience=10, mode='auto')
# model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1,
#           validation_data=(x_val, y_val),
#           callbacks=[early_stopping, tb_hist])

# #4. 평가예측
# loss, mse = model.evaluate(x_test, y_test, batch_size=1)
# print('mse : ', mse)


# x_prd=np.array([[401,402,403],[404,405,406],[407,408,409]])
# x_prd = np.transpose(x_prd)

# aaa=model.predict(x_prd, batch_size=1)
# print(aaa)

# # bbb = model.predict(x, batch_size=1)
# # print(bbb)

# #RMSE 구하기
# from sklearn.metrics import mean_squared_error

# y_predict = model.predict(x_test, batch_size=1)

# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print("RMSE : ", RMSE(y_test, y_predict))

# #R2 구하기
# from sklearn.metrics import r2_score
# r2_y_predict = r2_score(y_test, y_predict)
# print('R2 : ', r2_y_predict)
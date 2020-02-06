import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from keras import models, layers, initializers, losses, optimizers, metrics
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# 데이터 읽어 들이기
wine = pd.read_csv('./data/winequality-white.csv',
                   sep=";", encoding='utf-8')

# 데이터를 레이블과 데이터로 분리하기
y = wine['quality']
x = wine.drop('quality', axis=1)

print(x.shape)
print(y.shape)

# y 레이블 변경하기
newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist

label_encoder=LabelEncoder()
label_ids=label_encoder.fit_transform(y)

onehot_encoder=OneHotEncoder(sparse=False)
reshaped=label_ids.reshape(len(label_ids), 1)
onehot=onehot_encoder.fit_transform(reshaped)


x_train, x_test, y_train, y_test = train_test_split(
    x, onehot, test_size= 0.2, train_size = 0.8)

# 모델
model = Sequential()
model.add(Dense(32, input_shape= (11,), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))

#모델 컴파일
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

#모델 실행
model.fit(x_train, y_train, epochs=100, batch_size=10)

y_pred = model.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_pred))

# # 모델 구성
# model = RandomForestClassifier()

# # 훈련
# model.fit(x_train, y_train)

# '''
# keras에는 model fit, evaluate, predict가 있지만
# sklearn에는 evaluate가 없고 score가 있다
# '''

# # 평가 예측 
# aaa = model.score(x_test, y_test) # loss를 뺀 accuracy만 제공
# print('aaa : ', aaa)

# y_pred = model.predict(x_test)
# print('정답률 : ', accuracy_score(y_test, y_pred))

# print(classification_report(y_test, y_pred))
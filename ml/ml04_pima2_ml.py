# LinearSVC, KNeighborsClassifier

from keras.models import Sequential
from keras.layers import Dense
import numpy
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dataset = numpy.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

# 모델의 설정
model = LinearSVC()
# model.add(Dense(12, input_dim=8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# 모델 실행
model.fit(X,Y)
y_predict = model.predict(X)

# 결과 출력
print("acc = ", accuracy_score(Y, y_predict))
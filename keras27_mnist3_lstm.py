from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28*28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28*28, 1).astype('float32') / 255

from keras.utils import np_utils

# One-Hot Encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, LSTM

model = Sequential()
model.add(LSTM(32,input_shape=(28*28, 1)))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stopping = EarlyStopping(monitor='loss', patience=20)

model.fit(x_train, y_train, validation_split=0.2, batch_size=8, epochs=100, verbose=1, callbacks=[early_stopping])

acc = model.predict(x_test, y_test)
print(acc)


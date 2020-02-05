from keras.datasets import cifar10

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)
print(x_test.shape)

x_train = x_train.reshape(x_train.shape[0], 32*32*3).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 32*32*3).astype('float32') / 255

from keras.utils import np_utils

# One-Hot Encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

model = Sequential()
model.add(Dense(512, input_shape=(32*32*3, )))
# model.add(Conv2D(32, (2, 2), padding='same'))
# model.add(MaxPool2D((2, 2)))
# model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stopping = EarlyStopping(monitor='loss', patience=10)

model.fit(x_train, y_train, validation_split=0.2, batch_size=512, epochs=50, callbacks=[early_stopping])

loss, acc = model.evaluate(x_test, y_test, batch_size=512)
print("loss: ", loss)
print("acc: ", acc)



from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28 * 28).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28 * 28).astype('float32') / 255

from keras.utils import np_utils

# One-Hot Encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()
model.add(Dense(32, input_shape=(28 * 28, )))
# model.add(Conv2D(32, (2, 2), padding='same'))
# model.add(MaxPool2D((2, 2)))
# model.add(Flatten())
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

early_stopping = EarlyStopping(monitor='loss', patience=20)

history = model.fit(x_train, y_train, validation_split=0.2, batch_size=8, epochs=100, callbacks=[early_stopping])

model.evaluate(x_test, y_test, batch_size=1)

print(history.history.keys())

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model loss, accuracy')
plt.ylabel(['loss, acc'])
plt.xlabel(['epoch'])
plt.legend(['train loss', 'test loss', 'train acc', 'test acc'])
plt.show()
'''
loss, acc = model.predict(x_test, y_test)
print("loss: ", loss)
print("acc: ", acc)
'''

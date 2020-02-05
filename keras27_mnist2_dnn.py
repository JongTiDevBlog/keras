#!/usr/bin/env python
# coding: utf-8

# In[23]:


from keras.datasets import mnist


# In[24]:


from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping
import numpy as np


# In[25]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[26]:


x_train


# In[27]:


y_train


# In[28]:


x_train.shape


# In[29]:


y_train.shape


# In[30]:


x_train = x_train.reshape(x_train.shape[0], 28*28).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32') / 255


# In[31]:


print(type(x_train))


# In[32]:


from keras.utils import np_utils


# In[33]:


# One-Hot Encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


# In[34]:


y_train.shape


# In[35]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten


# In[36]:


import keras.backend.tensorflow_backend as K
with K.tf.device('/gpu:0'):

    model = Sequential()
    model.add(Dense(32,input_shape=(28*28, )))
    model.add(Dense(32))
    model.add(Dense(10, activation='softmax'))


# In[37]:


model.summary()


# In[38]:


# Use Multiple GPU
# from keras.utils import multi_gpu_model
# model = multi_gpu_model(model, gpus=4)


# In[39]:


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


# In[40]:


early_stopping = EarlyStopping(monitor='loss', patience=20)


# In[41]:


model.fit(x_train, y_train, validation_split=0.2, batch_size=8, epochs=100, verbose=1, callbacks=[early_stopping])


# In[ ]:


acc = model.predict(x_test, y_test)
print(acc)


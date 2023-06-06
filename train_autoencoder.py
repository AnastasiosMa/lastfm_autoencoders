#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 13:29:28 2023

@author: anmavrol
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,RepeatVector,TimeDistributed,LSTM,Input,\
Conv2D,MaxPooling2D,Conv2DTranspose
from tensorflow.keras.callbacks import EarlyStopping
#%% load mfccs
with open('mfccs_rock.txt','r') as f:
    data = json.load(f)    

x = np.array(data['mfcc'])
dims = x.shape

x_train, x_test = train_test_split(x,test_size = 0.2)
#%% Create model
input_layer = Input(shape=(300,40))

encoder = LSTM(128, activation='relu')(input_layer)
# define reconstruct decoder
decoder = RepeatVector(dims[1])(encoder)
decoder = LSTM(128, activation='relu', return_sequences=True)(decoder)
decoder = TimeDistributed(Dense(dims[2]))(decoder)

model = Model(inputs=input_layer, outputs=decoder)
model.compile(optimizer='adam', loss='mse',metrics=['mse'])

#encoder_model = Model(inputs=input_layer, outputs=encoder)
encoder_model = Model(inputs=model.inputs, outputs=model.layers[1].output)
callbacks = [EarlyStopping(monitor='loss', min_delta=0, patience=8, verbose=1, mode='auto')]

history = model.fit(x_train,x_train,validation_data=(x_test, x_test),batch_size=32,epochs=1000,verbose=1,callbacks = callbacks)
model.save('LSTM_AE.h5')

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

#%%
cnn_input_layer = Input(shape=(300,40,1))
conv1 = Conv2D(64,(3,3),padding='same',activation = 'relu')(cnn_input_layer)
pool1 = MaxPooling2D((2,2),strides=2)(conv1)
conv2 = Conv2D(64,(3,3),padding='same',activation = 'relu')(pool1)
pool2 = MaxPooling2D((2,2),padding='same')(conv2)

transp1 = Conv2DTranspose(64, (3, 3), padding='same', activation='relu', strides=2)(pool2)
transp2 = Conv2DTranspose(64, (3, 3), padding='same', activation='relu', strides=2)(transp1)
output_layer = Conv2D(1, (3, 3), padding='same',activation='sigmoid')(transp2)
cnn_ae = Model(cnn_input_layer,output_layer)
cnn_ae.compile(loss='mse', optimizer='adam', metrics=['mse'])
history = model.fit(x_train,x_train,batch_size=32,epochs=1000,verbose=1,validation_data=(x_test, x_test),callbacks = callbacks)
cnn_ae.summary()
cnn_ae.save('CNN_AE.h5')
#global_pool = GlobalAveragePooling2D()(conv2)
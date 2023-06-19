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
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,RepeatVector,TimeDistributed,LSTM,Input,\
Conv2D,MaxPooling2D,Conv2DTranspose
from tensorflow.keras.callbacks import EarlyStopping
#%% load mfccs
with open('data/mfccs_rock.txt','r') as f:
    data = json.load(f)    

x = np.array(data['mfcc'])
dims = x.shape

scaler = MinMaxScaler()

for k in range(x.shape[1]):
    x[:,k,:] = scaler.fit_transform(x[:,k,:])
    
x_train, x_test = train_test_split(x, test_size=0.2)

callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')]
#%% Create model
input_layer = Input(shape=(300,40))

encoder = LSTM(128, activation='relu')(input_layer)
# define reconstruct decoder
decoder = RepeatVector(dims[1])(encoder)
decoder = LSTM(128, activation='relu', return_sequences=True)(decoder)
decoder = TimeDistributed(Dense(dims[2]))(decoder)

model = Model(inputs=input_layer, outputs=decoder)
model.compile(optimizer='adam', loss='mse',metrics=['mse'])

history = model.fit(x,x,validation_data=(x, x),batch_size=32,epochs=1000,verbose=1,callbacks = callbacks)
model.save('models/LSTM_AE.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#%% CNN AE
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
history = cnn_ae.fit(x,x,batch_size=32,epochs=1000,verbose=1,validation_data=(x_test, x_test),callbacks = callbacks)
cnn_ae.summary()
cnn_ae.save('models/CNN_AE.h5')

#%% Non symmetric cnn ae
#encoder
cnn_input_layer = Input(shape=(300,40,1))
conv1 = Conv2D(256,(6,3),padding='same',activation = 'relu')(cnn_input_layer)
pool1 = MaxPooling2D((3,2),strides=(3,2))(conv1)
conv2 = Conv2D(128,(4,2),padding='same',activation = 'relu')(pool1)
pool2 = MaxPooling2D((4,2),strides=(4,2))(conv2)
conv3 = Conv2D(64,(5,2),padding='same',activation = 'relu')(pool2)
pool3 = MaxPooling2D((5,2),strides=(5,2))(conv3)
cnn_encoder = Model(cnn_input_layer,pool3)
#cnn_encoder.summary()

#decoder
transp1 = Conv2DTranspose(64, (5,2), padding='same', activation='relu', strides=(5,2))(pool3)
transp2 = Conv2DTranspose(128, (4, 2), padding='same', activation='relu', strides=(4,2))(transp1)
transp3 = Conv2DTranspose(256, (3, 2), padding='same', activation='relu', strides=(3,2))(transp2)
output_layer = Conv2D(1, (3, 3), padding='same',activation='sigmoid')(transp3)
cnn_ae = Model(cnn_input_layer,output_layer)
cnn_ae.summary()
cnn_ae.compile(loss='mse', optimizer='adam', metrics=['mse'])
history = cnn_ae.fit(x,x,batch_size=32,epochs=1000,verbose=1,validation_data=(x_test, x_test),callbacks = callbacks)
cnn_ae.save('models/CNN_AE_ASSYMETRIC.h5')
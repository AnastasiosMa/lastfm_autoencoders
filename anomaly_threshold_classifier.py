#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 14:44:36 2023

@author: anmavrol
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
#%% Load models and data
lstm_model = tf.keras.models.load_model('models/LSTM_AE.h5')
lstm_encoder_model = Model(inputs=lstm_model.inputs, outputs=lstm_model.layers[1].output)

cnn_model = tf.keras.models.load_model('models/CNN_AE.h5')
cnn_encoder_model = Model(inputs=cnn_model.inputs, outputs=cnn_model.layers[1].output)

#load spotify genres
with open('data/genre_data.txt','r') as f:
    spotify_genre = json.load(f)
    
# load mfccs
with open('data/mfccs_rock.txt','r') as f:
    data = json.load(f)    

x = np.array(data['mfcc'])
dims = x.shape

scaler = MinMaxScaler()

for k in range(x.shape[1]):
    x[:,k,:] = scaler.fit_transform(x[:,k,:])
#%% Calculate reconstruction loss
x_predictions = cnn_model.predict(x)
cnn_mse = np.sum(np.sum(np.power(x - np.squeeze(x_predictions), 2), axis=1), axis=1)#/(x.shape[1]*x.shape[2])

x_predictions = lstm_model.predict(x)
lstm_mse = np.sum(np.sum(np.power(x - np.squeeze(x_predictions), 2), axis=1), axis=1)/(x.shape[1]*x.shape[2])

#%% Plots
def plot_mse(mse):
    fig, ax = plt.subplots()
    ax.plot(mse,marker='o', ms=3.5,linestyle='')
    plt.xlabel('Music tracks')
    plt.ylabel('Reconstruction error')
    thx = np.mean(mse) + 3*np.std(mse)
    ax.hlines(thx,ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='3 std threshold')
    ax.legend()
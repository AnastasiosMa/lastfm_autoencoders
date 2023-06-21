#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 14:44:36 2023

@author: anmavrol
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import pandas as pd
#%% Load models and data
lstm_model = tf.keras.models.load_model('models/LSTM_AE.h5')
lstm_encoder_model = Model(inputs=lstm_model.inputs, outputs=lstm_model.layers[1].output)

cnn_model = tf.keras.models.load_model('models/CNN_AE.h5')
cnn_encoder_model = Model(inputs=cnn_model.inputs, outputs=cnn_model.layers[4].output)

cnn_assymetric_model = tf.keras.models.load_model('models/CNN_AE_ASSYMETRIC.h5')

top_chart_tracks = pd.read_csv('data/top_charts_track_info.csv')       

with open('data/mfccs/top_charts_mfccs.txt','r') as f:
    top_chart_mfccs = json.load(f)

#remove tracks with no audio
x = np.array(top_chart_mfccs['mfcc'])
filename = list(top_chart_mfccs['labels'])
top_chart_ids = list(top_chart_tracks.track_id)
track_idx = [top_chart_ids.index(track[:-4]) for track in filename]
top_chart_tracks = top_chart_tracks.iloc[track_idx,:]
top_chart_tracks.index = range(len(top_chart_tracks))

#find invalid idxs
tracks_to_remove = top_chart_tracks.index[top_chart_tracks.isrock == '2'].tolist()
x = np.delete(x,tracks_to_remove,0)
top_chart_tracks = top_chart_tracks.drop(tracks_to_remove,0)
y  = np.array(top_chart_tracks['isrock'])

y = np.array(y=='True',dtype=int)*1

scaler = MinMaxScaler()
dims = x.shape
for k in range(x.shape[1]):
    x[:,k,:] = scaler.fit_transform(x[:,k,:])
#%% Load ae models
x_predictions = cnn_model.predict(x)
cnn_mse = np.sum(np.sum(np.power(x - np.squeeze(x_predictions), 2), axis=1), axis=1)#/(x.shape[1]*x.shape[2])

x_predictions = lstm_model.predict(x)
lstm_mse = np.sum(np.sum(np.power(x - np.squeeze(x_predictions), 2), axis=1), axis=1)/(x.shape[1]*x.shape[2])

x_predictions = cnn_assymetric_model.predict(x)
cnn_assymetric_mse = np.sum(np.sum(np.power(x - np.squeeze(x_predictions), 2), axis=1), axis=1)/(x.shape[1]*x.shape[2])
#%% Fit logistic regression to mse to estimate anomaly threshold
def mse_analysis(mse,y):
    f, ax1 = plt.subplots()
    ax1.plot(mse,marker='o', ms=3.5,linestyle='')
    ax1.axes.set_xlabel('Music tracks')
    ax1.axes.set_ylabel('Reconstruction error')
    thx = np.mean(mse) + 3*np.std(mse)
    ax1.hlines(thx,ax1.get_xlim()[0], ax1.get_xlim()[1], colors="r", zorder=100, label='3 std threshold')
    ax1.legend()
    
    error_df = pd.DataFrame({'Reconstruction_error': mse,
                            'True_class': y})
    groups = error_df.groupby('True_class')
    
    f, ax2 = plt.subplots()
    for name, group in groups:
        ax2.plot(group.index, group.Reconstruction_error,\
                marker='o', ms=3.5,linestyle='',label= "Rock" if name == 1 else "Non-Rock")
    plt.xlabel('Music tracks')
    plt.ylabel('Reconstruction error')
    thx = np.mean(mse) + 3*np.std(mse)
    ax2.hlines(thx,ax2.get_xlim()[0], ax2.get_xlim()[1], colors="r", zorder=100, label='3 std threshold')
    ax2.legend()
    plt.show()
        
    scaler = MinMaxScaler()
    x = scaler.fit_transform(mse.reshape(-1,1))
    x = 1-x
    reg = LogisticRegression().fit(x,y)
    y_pred = reg.predict(x)
    acc = reg.score(x,y)
    print(acc)

mse_analysis(cnn_assymetric_mse,y)
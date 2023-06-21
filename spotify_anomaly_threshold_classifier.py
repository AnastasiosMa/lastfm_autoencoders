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

#load spotify genres
with open('data/spotify_genre_data.txt','r') as f:
    spotify_genre = json.load(f)
    
# load mfccs
with open('data/mfccs/mfccs_rock.txt','r') as f:
    data = json.load(f)    

x = np.array(data['mfcc'])
dims = x.shape

scaler = MinMaxScaler()

for k in range(x.shape[1]):
    x[:,k,:] = scaler.fit_transform(x[:,k,:])
#%% Plot spotify genres
#flatten list
genre_list = [element for genre in spotify_genre['genres'] for element in genre]
genre_count = Counter(genre_list)
N = len(genre_count.keys())

idx=np.argsort(list(genre_count.values()))
sorted_values = np.array(list(genre_count.values()))[idx]
sorted_keys = list(genre_count.keys())
sorted_keys = [sorted_keys[index] for index in idx]

fig, ax = plt.subplots()
ax.plot(sorted(list(genre_count.values()),reverse=True))
plt.xlabel('Spotify genres (N = {})'.format(N))
plt.ylabel('Count')


f, ax = plt.subplots(figsize=(5, 12))
sns.barplot(x = sorted_values[-30:-1],y = sorted_keys[-30:-1])
plt.xlabel('Count')
plt.ylabel('Top 30 Genres')

# Convert genres to ground truth
y = np.array([1 if any('rock' in s for s in genre) else 0 for genre in spotify_genre['genres']]) 
#%% Load ae models
x_predictions = cnn_model.predict(x)
cnn_mse = np.sum(np.sum(np.power(x - np.squeeze(x_predictions), 2), axis=1), axis=1)#/(x.shape[1]*x.shape[2])

x_predictions = lstm_model.predict(x)
lstm_mse = np.sum(np.sum(np.power(x - np.squeeze(x_predictions), 2), axis=1), axis=1)/(x.shape[1]*x.shape[2])

#%% Spotify genre comparison analysis
def mse_analysis(mse,y):
    
    true_mse=mse[y==1]
    false_mse=mse[y==0]
    # Crossvalidation
    kFold = int(round(len(y)/(len(y)-np.sum(y)),0))
    cv = KFold(n_splits=kFold,shuffle=True)
    
    for k, (train_samples,test_samples) in enumerate(cv.split(true_mse)):
    
        x_fold = np.concatenate((true_mse[test_samples],false_mse))
        y_fold = np.concatenate((np.ones(len(test_samples)),np.zeros(len(false_mse))))
        f, ax1 = plt.subplots()
        ax1.plot(x_fold,marker='o', ms=3.5,linestyle='')
        ax1.axes.set_xlabel('Music tracks')
        ax1.axes.set_ylabel('Reconstruction error')
        thx = np.mean(x_fold) + 3*np.std(x_fold)
        ax1.hlines(thx,ax1.get_xlim()[0], ax1.get_xlim()[1], colors="r", zorder=100, label='3 std threshold')
        ax1.legend()
        
        error_df = pd.DataFrame({'Reconstruction_error': x_fold,
                                'True_class': y_fold})
        groups = error_df.groupby('True_class')
        
        f, ax2 = plt.subplots()
        for name, group in groups:
            ax2.plot(group.index, group.Reconstruction_error,\
                    marker='o', ms=3.5,linestyle='',label= "Rock" if name == 1 else "Non-Rock")
        plt.xlabel('Music tracks')
        plt.ylabel('Reconstruction error')
        thx = np.mean(x_fold) + 3*np.std(x_fold)
        ax2.hlines(thx,ax2.get_xlim()[0], ax2.get_xlim()[1], colors="r", zorder=100, label='3 std threshold')
        ax2.legend()
        plt.show()
        
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x_fold.reshape(-1,1))
        x = 1-x
        reg = LogisticRegression().fit(x,y_fold)
        y_pred = reg.predict(x)
        acc = reg.score(x,y_fold)
        print(acc)

mse_analysis(cnn_mse,y)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:34:44 2023

@author: anmavrol
"""

import librosa
import pandas as pd
import numpy as np
import os
import math
import csv
import warnings
import json
#warnings.filterwarnings("ignore")
dataset_path = 'data/track_previews/'
#%% Extract mfccs function
def preprocess(dataset_path,num_mfcc=40,n_fft=4096,hop_length=2048):
    data = {"labels":[],"mfcc":[]}
    sample_rate = 22050
    
    for idx, filename in enumerate(os.listdir(dataset_path)):
        print(idx)
        print('Track name ', filename)
        
        try:
            y,sr = librosa.load(dataset_path + filename,sr = sample_rate)
        except:
            continue
        mfcc = librosa.feature.mfcc(y=y,sr=sample_rate,n_mfcc = num_mfcc, n_fft = n_fft,
                                    hop_length = hop_length)
        mfcc = mfcc.T
        if len(mfcc) > 300:
            mfcc = mfcc[:300,:]
        data['mfcc'].append(mfcc.tolist())
        data['labels'].append(filename)
    return data
#%% extract mfccs and save data
mfcc_data = preprocess(dataset_path)

with open('mfccs_rock.txt','w') as f:
    json.dump(mfcc_data, f)

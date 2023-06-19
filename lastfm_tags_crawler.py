#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 12:21:03 2023

@author: anmavrol
"""

import requests
import time
import pandas as pd
import os
import itertools
import time
import pickle
import csv

#%% load data
with open('data/lastfm_authentication.txt') as f:
    credentials = f.readlines()
_ipkey = credentials[0]
_secret = credentials[1]

#%% last fm class
class LastFMMethodTag:
    def __init__(self, client_id,method,tag,page):
        # First header and parameters needed to require an access token.
        self.base_url = "http://ws.audioscrobbler.com/2.0/?"
        self.url = self.base_url + "method=" + method + "&tag=" + tag + "&api_key=" + client_id + "&format=json"
        self.params = {'page':page}
        self.response = self.api_call(self.url,self.params)
        
    def api_call(self,url,params):
        url = url.encode(encoding = 'UTF-8', errors = 'strict')
        return requests.get(url,params)
#%%
track_name = []
artist_name = []
for i in range(100):
    response = LastFMMethodTag(_ipkey,'tag.gettoptracks','rock',str(i)).response
    if response.status_code==200:
        print(i)
        response = response.json()
        track_name.extend([x['name'] for x in response['tracks']['track']])
        artist_name.extend([x['artist']['name'] for x in response['tracks']['track']])
    else:
        print('Error request:{},{}'.format(i,response.status_code))
fields = ['track_name','artist_name']
data = [track_name,artist_name]

data = [[i,j] for i, j in zip(track_name, artist_name)]


with open('data/track_info.csv','w') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(data)
    
#%% GET CHARTS
class LastFMChartMethod:
    def __init__(self, client_id,method,page):
        # First header and parameters needed to require an access token.
        self.base_url = "http://ws.audioscrobbler.com/2.0/?"
        self.url = self.base_url + "method=" + method + "&api_key=" + client_id + "&format=json"
        self.params = {'page':page}
        self.response = self.api_call(self.url,self.params)
        
    def api_call(self,url,params):
        url = url.encode(encoding = 'UTF-8', errors = 'strict')
        return requests.get(url,params)
    
track_name = []
artist_name = []
isrock = []
for i in range(100):
    response = LastFMChartMethod(_ipkey,'chart.getTopTracks',str(i+1)).response
    if response.status_code==200:
        print('PAGE NUMBER: ' + str(i))
        response = response.json()
        track_name.extend([x['name'] for x in response['tracks']['track']])
        artist_name.extend([x['artist']['name'] for x in response['tracks']['track']])
        
        for k in range(len(response['tracks']['track'])):
            url = 'http://ws.audioscrobbler.com/2.0/?method=track.gettoptags' + '&api_key=' + _ipkey + \
            '&artist=' + response['tracks']['track'][k]['artist']['name'].replace(' ','+') + \
                '&track=' + response['tracks']['track'][k]['name'].replace(' ','+')+ '&autocorrect=1' +"&format=json"
            track = requests.get(url.encode(encoding = 'UTF-8', errors = 'strict')).json()
            
            try:
                isrock.append(any([1 if tag['name'] == 'rock' else 0 for tag in track['toptags']['tag'][:20]]))
            except:
                isrock.append(2)
            print(response['tracks']['track'][k]['name'] + ' ' + response['tracks']['track'][k]['artist']['name'])
            print(isrock[-1])
            
    else:
        print('Error request:{},{}'.format(i,response.status_code))
fields = ['track_name','artist_name','isrock']
data = [track_name,artist_name,isrock]

data = [[i,j] for i, j in zip(track_name, artist_name, isrock)]


with open('data/classification_sample.csv','w') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(data)
    

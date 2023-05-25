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
class Connection:
    """
    Instantiate a connection with last Fm API
    """
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
    response = Connection(_ipkey,'tag.gettoptracks','rock',str(i)).response
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
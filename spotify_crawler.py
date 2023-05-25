#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:38:13 2023

@author: anmavrol
"""
import requests
import base64
import os
import pandas as pd
import numpy as np
from IPython.core.debugger import Pdb
ipdb = Pdb()

# The root directory to save your music previews.
save_loc = r'/data/track_previews'

#%% SPOTIFY IDENTIFICATION
with open ('data/spotify_user_authentication.txt') as f:
    user_password = f.readlines()
_id = user_password[0][0:-1]
_secret = user_password[1][0:-1]
#%% SPOTIFY CONNECTION FUNCTION
class Connection:
    """
    Class' object instantiates a connection with spotify. When the connection is alive, queries are made with the query_get
    method.
    """
    def __init__(self, client_id, secret):
        # First header and parameters needed to require an access token.
        param = {"grant_type": "client_credentials"}
        header = {"Authorization": "Basic {}".format(
            base64.b64encode("{}:{}".format(client_id, secret).encode("ascii")).decode("ascii")),
                  'Content-Type': 'application/x-www-form-urlencoded'}
        self.token = requests.post("https://accounts.spotify.com/api/token", param, headers=header).json()["access_token"]
        self.header = {"Authorization": "Bearer {}".format(self.token)}
        self.base_url = "https://api.spotify.com"

    def query_get(self, query, params=None):
        """
        
        :param query: (str) URL coming after example.com
        :param params: (dict)
        :return: (json) 
        """
        return requests.get(self.base_url + query, params, headers=self.header).json()
#%% GET SPOTIFY VARIABLES FUNCTIONS
# Search for IDs through v1/search
def get_id(track_name, artist_name):
    """
    If id is missing try to find through search
    """
    track_id=[]
    if isinstance(artist_name, (str)):
       query_track = dict(q = track_name + ' ' + artist_name, type = "track", limit = 50) 
    else:
       query_track = dict(q = track_name, type = "track", limit = 50)       
    search = conn.query_get('/v1/search/',query_track)
    for i in range(len(search['tracks']['items'])):
        artist = search['tracks']['items'][i]['artists'][0]['name']
        name = search['tracks']['items'][i]['name']
        if artist_name.lower()==artist.lower() and track_name.lower()==name.lower():
           track_id = search['tracks']['items'][i]['id'] 
           break
    return(track_id)
#%% LOAD ESM FILE
#path=r'~/Desktop/mupsychapp/data/MuPsych Data main file.xlsx'
#data = pd.read_excel (path,sheet_name = 'Music data',header=1,engine='openpyxl')
path='data/track_info.csv'
data = pd.read_csv(path)
#%% GET SPOTIFY VARIABLES
conn = Connection(_id, _secret)
#%%
for i in range(len(data.iloc[:,1])):
    if i%300==0:
       conn = Connection(_id, _secret) 
    print(i)
    track_name = data.loc[i,'track_name']
    artist_name = data.loc[i,'artist_name']
    track_id = get_id(track_name, artist_name)
    if track_id:
        print('trackid_found')
        
    else:
       track_id = float('nan')
    if isinstance(track_id, (int, str)):
       data.loc[i,'track_id'] = track_id
    else:
       data.loc[i,'track_id']=''     
#%% GET PREVIEWS FUNCTION
def download_track(track_id,artist,name,save_loc,url):
    os.makedirs(save_loc, exist_ok=True)
    f = os.path.join(save_loc, "{}.mp3".format(track_id))
    if not os.path.isfile(f):
           r = requests.get(url)
           print("Saving {}-{}.mp3".format(artist, name))
           print("ID: " + track_id)
           with open(f, "wb") as f:
                        f.write(r.content)
    else:
           print("file already exists:{}-{}".format(artist, name))
#%% GET PREVIEWS    
conn = Connection(_id, _secret)
track_ids = data['trackID']
#track_ids = list(set(track_ids)) #store unique ID's
missing_track=[]
missing_previews = 0
found_through_trackname = 0
idx_trackname = []
found_through_artist = 0
idx_artist = []

for i in range(len(track_ids)):
    # get track preview_urls
    if i%300==0:
       conn = Connection(_id, _secret) 
    print(i)
    track = conn.query_get('/v1/tracks/' + data.loc[i,'trackID'])
    if not 'error' in dict.keys(track):
         track_name = track["name"]
         #name = re.sub("\W", "_", name)
         artist_name = track["artists"][0]["name"]
         #artist = re.sub("\W", "_", artist)
         url = track["preview_url"]
         print("Searching url for {}-{}".format(artist_name, track_name))
         if url:
            download_track(track_ids[i],artist_name,track_name,save_loc,url)
            print("url found")
         else:
              print("could not find url")
    elif len(track_ids[i])==0:
         print("NanTrack")                
    else:
        print("Invalid ID")


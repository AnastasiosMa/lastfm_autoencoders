#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:38:13 2023

@author: anmavrol
"""
import requests
import base64
import os
import json
from IPython.core.debugger import Pdb
ipdb = Pdb()

# The root directory of music previews.
dataset_path = 'data/track_previews/'
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
# Get genres from ID
def get_genres(track_id):
    track_obj = conn.query_get('/v1/tracks/' + track_id)
    artist_obj = conn.query_get('/v1/artists/' + track_obj['artists'][0]['id'])
    genres = artist_obj['genres']
    return(genres)
#%% GET GENRES
genre_data = {'genres':[],'labels':[]}
conn = Connection(_id, _secret)
for idx, filename in enumerate(os.listdir(dataset_path)):
    if idx%300==0:
       conn = Connection(_id, _secret) 
    print(idx)
    genres = get_genres(filename[:-4])
    genre_data['genres'].append(genres)
    genre_data['labels'].append(filename)
    
with open('data/genre_data.txt','w') as f:
    json.dump(genre_data, f)
    


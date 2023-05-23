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

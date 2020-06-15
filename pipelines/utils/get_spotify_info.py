#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
import time
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Now try to retrieve spotify information for each artist/song    

from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

def get_track_ids(client, data, num_tracks=None, turn_update=None):
    
    for i, row in data.iterrows():
        artist=row['artist']
        song=row['song']
        q = 'artist:'+artist + ' ' + 'track:'+song
        result = client.search(q)
        items = result['tracks']['items']
        if not items:
            continue
        popularities = [item['popularity'] for item in items]
        most_popular = items[popularities.index(max(popularities))]
        data.loc[(data.artist==artist) & (data.song==song),'spotify_ID']=most_popular['id']
        if i == num_tracks:
            break
        if turn_update and (i+1)%turn_update==0:
            print('Finished track {}'.format(i+1))
    
def get_audio_features(client, data, verbose=False):
    
    ids = data.spotify_ID.dropna()
    
    audio_feature_data = pd.DataFrame([],columns=['danceability', 'energy',
                                      'key', 'loudness', 'mode', 'speechiness',
                                      'acousticness', 'instrumentalness',
                                      'liveness', 'valence', 'tempo','id',
                                      'duration_ms', 'time_signature'])
    for i in range(0, len(ids), 50):
        new_features = client.audio_features(ids[i:i+50])
        for track in new_features:
            track.pop('type')
            track.pop('uri')
            track.pop('track_href')
            track.pop('analysis_url')
        audio_feature_data = audio_feature_data.append(new_features)
        if verbose:
            print('Done with tracks {} through {}'.format(i+1, i+50))
    
    return audio_feature_data
    
def get_track_genres(client, track_ids, verbose=False):
    data = pd.DataFrame([], columns=['spotify_ID', 'genres'])
    for i in range(0, len(track_ids), 20):
        tids = track_ids[i:i+20]
        tracks = client.tracks(tids)['tracks']
        artist_ids = [track['artists'][0]['id'] for track in tracks]
        artists = client.artists(artist_ids)['artists']
        genres = [artist['genres'] for artist in artists]
        new_data = [{'spotify_ID':tid, 'genres':genre} for tid,genre in zip(tids, genres)]
        data = data.append(new_data).dropna()
        if verbose:
            print('Finished fetching genres for tids {} through {}.'.format(i+1, i+20))
            print('New data shape: {}'.format(data.shape))
            
    return data
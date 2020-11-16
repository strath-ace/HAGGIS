import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import os
import pdb

def dotted_rhythms_8track(mus_proll):
    mus_proll = mus_proll.reshape(-1,84,8)
    n_beat = int(mus_proll.shape[0]/24)
    dot_beats = np.zeros(n_beat).astype(bool)
    for t in range(8):
        track_test = mus_proll[:,:,t]
        for i,b in enumerate(trange(0,track_test.shape[0]-23,24)):
            if dot_beats[i]:
                continue
            mus_beat = track_test[b:b+24,:]
            note_beat = np.where(mus_beat)
            if len(note_beat[0])>=24: 
                _,beat_count = np.unique(note_beat[1],return_counts=True)
                if np.any(beat_count==6) and np.any(beat_count==18):
                    dot_beats[i] = True
    return dot_beats


print('Loading data...')
their_data = np.load('lastfm_alternative_8b_phrase.npy')
print(np.shape(their_data))

db_theirs = dotted_rhythms_8track(their_data)

print('Their data:')
print(float(len(np.where(db_theirs)[0]))/float(len(db_theirs)))

del(their_data)

print('Loading data...')
our_data = np.load('final-balanced-scottish-midi-data.npy')
print(np.shape(our_data))

db_ours = dotted_rhythms_8track(our_data)

print('Our data:')
print(float(len(np.where(db_ours)[0]))/float(len(db_ours)))
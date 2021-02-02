# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2021 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------
"""Functions implementing the new scottish metrics"""

import numpy as np
import os
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import muspy
from music_funcs import mus_to_8track_array, f_track_converter

pentatonic = np.array([1,3,5,8,10])
pent_scales = [pentatonic.copy()]
for _ in range(11):
    pentatonic += 1
    pentatonic[pentatonic>12] -= 12
    pent_scales.append(pentatonic.copy())

def dotted_rhythms_8track(mus_proll):
    """Dotted rhythm metric applied to 8 track pianoroll"""
    mus_proll = mus_proll.reshape(-1,84,8)
    n_beat = int(mus_proll.shape[0]/24)
    dot_beats = np.zeros(n_beat).astype(bool)
    for t in range(8):
        track_test = mus_proll[:,:,t]
        for i,b in enumerate(range(0,track_test.shape[0]-23,24)):
            if dot_beats[i]:
                continue
            mus_beat = track_test[b:b+24,:]
            note_beat = np.where(mus_beat)
            if len(note_beat[0])>=24: 
                _,beat_count = np.unique(note_beat[1],return_counts=True)
                if np.any(beat_count==6) and np.any(beat_count==18):
                    dot_beats[i] = True
    return float(len(dot_beats[dot_beats]))/float(len(dot_beats))

def pent_notes_8track(mus_proll):
    """Pentatonic metric applied to 8 track pianoroll"""
    mus_proll = mus_proll.reshape(-1,4,96,84,8)
    mus_tracks = mus_proll[:,:,:,:,1:]

    all_pent_props = []
    all_count_props = []
    for t in mus_tracks:
        proll_inds = np.where(t)
        note_inds = proll_inds[3]%12
        all_note, c_note = np.unique(note_inds,return_counts=True)
        sum_note = np.sum(c_note)
        if sum_note==0:
            continue
        max_p = 0
        for p in pent_scales:
            note_mask = [n in p for n in all_note]
            max_p = np.max( (max_p, float(np.sum(c_note[note_mask]))/float(sum_note)) )
        all_pent_props.append(max_p)
        all_count_props.append(sum_note)
        
    return float(np.sum(np.array(all_pent_props)*np.array(all_count_props)))/float(np.sum(all_count_props))

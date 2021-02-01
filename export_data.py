# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2021 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------

import muspy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm, trange
import os

inst_data = ['piano']*8 + ['perc']*8 + ['organ']*8 + ['guitar']*8 + ['bass']*8 + ['strings']*8 + ['ensemble']*8 + ['brass']*8 + ['reed']*8 + ['pipe']*8 + ['synth_lead']*8 + ['synth_pad']*8 + ['synth_eff']*8 + ['ethnic']*8 + ['percussive']*8 + ['sound_eff']*8

def describe_data(mus):
    """Returns various characteristics of a music object from a single file"""
    title = mus.metadata.source_filename[:-4]
    mus = mus.adjust_resolution(target=24)
    n_track = len(mus)
    mus_piano = muspy.to_pianoroll_representation(mus)
    
    ind_nz = np.nonzero(mus_piano)
    note_min = np.min(ind_nz[1])
    note_max = np.max(ind_nz[1])
    mus_nz = mus_piano[ind_nz[0][0]:ind_nz[0][-1]+1,:]
    
    inst_nos = []
    inst_names = []
    n_drum = 0
    for track in mus:
        if track.is_drum:
            n_drum += 1
            inst_nos.append(-1)
            inst_names.append('drums')
        else:
            inst_nos.append(track.program+1)
            inst_names.append(inst_data[track.program])
    
    t_sig = '{}/{}'.format(mus.time_signatures[0].numerator,mus.time_signatures[0].denominator)
    tempo = mus.tempos[0].qpm
    return title, n_track, note_min, note_max, mus_nz.shape[0], inst_nos, inst_names, n_drum, t_sig, tempo


def describe_tracks(mus):
    """Returns a dictionary of characteristics of each track of a music object from a single file"""
    t_data = {'min_note':[], 'max_note':[], 'mean_note':[], 'n_note':[], 'poly_ratio':[], 'min_dur':[], 'max_dur':[], 'mean_dur':[], 'program':[], 'inst':[], 'new_inst':[]}
    for t in mus:
        t_piano = muspy.to_pianoroll_representation(muspy.Music(tracks=[t]))
        ind_nz = np.nonzero(t_piano)
        
        t_data['min_note'].append(np.min(ind_nz[1]))
        t_data['max_note'].append(np.max(ind_nz[1]))
        t_data['mean_note'].append(np.round(np.mean(ind_nz[1]),2))
        t_data['n_note'].append(len(t.notes))
        
        un_time, c_time = np.unique(ind_nz[0],return_counts=True)
        r_poly = float(len(np.where(c_time>1)[0]))/float(len(un_time))
        t_data['poly_ratio'].append(r_poly)
        
        note_durs = [float(n.duration)/float(mus.resolution) for n in t.notes]
        t_data['min_dur'].append(np.min(note_durs))
        t_data['max_dur'].append(np.max(note_durs))
        t_data['mean_dur'].append(np.round(np.mean(note_durs),3))
        
        t_data['program'].append(t.program+1)
        if t.is_drum:
            t_data['inst'].append('drum')
            t_data['new_inst'].append(0)
        else:
            t_data['inst'].append(inst_data[t.program])
            t_data['new_inst'].append(track_convert[t.program+1])
            
    return t_data


def balance_new_tracks(mus):
    """
    Apply heuristics to tracks in mus to assign new tracks
    ...

    Parameters
    ----------
    mus : muspy.Music
     Music to be converted

    Returns
    -------
    new_inst_nos : list
        For each track in mus, the corresponding new track to which it is assigned
    n_new : list
        Number of old tracks in each of the 8 new tracks
    """
    new_inst_nos = []
    for t in mus:
        if t.is_drum:
            new_inst_nos.append(0)
        else:
            new_inst_nos.append(track_convert[t.program+1])
    new_inst_nos = np.array(new_inst_nos)
    
    def get_n_new():
        """Number of old tracks in each new track"""
        n_new = []
        for i in range(8):
            n_new.append(len(np.where(new_inst_nos==i)[0]))
        return np.array(n_new)
    n_new = get_n_new()
    
    if np.any(np.array(n_new)>inst_max):
        t_info = pd.DataFrame(describe_tracks(mus))
        def update_t():
            t_info.new_inst = new_inst_nos
            
        if n_new[1]>inst_max:
            ind_bass = t_info.loc[(t_info.new_inst==1)&(t_info.mean_note<50)].index
            new_inst_nos[ind_bass] = 3
        n_new = get_n_new()
        update_t()
        
        if n_new[1]>inst_max:
            ind_melody = t_info.loc[(t_info.new_inst==1)&(t_info.mean_note>60)&(t_info.n_note>100)].index
            new_inst_nos[ind_melody[0]] = 4
        n_new = get_n_new()
        update_t()
        
        for _ in range(3):
            if n_new[1]>inst_max:
                empty_inst = np.where(n_new==0)[0]
                empty_inst = empty_inst[empty_inst>0]
                ind_non_poly = t_info.loc[(t_info.new_inst==1)&(t_info.poly_ratio<0.2)].index
                if len(empty_inst)>0 and len(ind_non_poly)>0:
                    new_inst_nos[ind_non_poly[0]] = empty_inst[0]
            n_new = get_n_new()
            update_t()
        
        if n_new[5]>inst_max:
            ind_bass = t_info.loc[(t_info.new_inst==5)&(t_info.mean_note<50)].index
            new_inst_nos[ind_bass] = 3
        n_new = get_n_new()
        update_t()
        
        if n_new[5]>inst_max:
            ind_melody = t_info.loc[(t_info.new_inst==5)&(t_info.mean_note>60)&(t_info.n_note>100)].index
            new_inst_nos[ind_melody] = 4
        n_new = get_n_new()
        update_t()
        
        for _ in range(3):
            if n_new[4]>inst_max:
                empty_inst = np.where(n_new==0)[0]
                empty_inst = empty_inst[empty_inst>0]
                ind_non_poly = t_info.loc[(t_info.new_inst==4)&(t_info.poly_ratio==0)].index
                if len(empty_inst)>0 and len(ind_non_poly)>0:
                    new_inst_nos[ind_non_poly[0]] = empty_inst[0]
            n_new = get_n_new()
            update_t()
            
        if n_new[2]>inst_max:
            ind_bass = t_info.loc[(t_info.new_inst==2)&(t_info.mean_note<50)].index
            new_inst_nos[ind_bass] = 3
        n_new = get_n_new()
        update_t()
        
        if n_new[2]>inst_max:
            ind_melody = t_info.loc[(t_info.new_inst==2)&(t_info.mean_note>60)&(t_info.n_note>100)].index
            if n_new[4]<2:
                new_inst_nos[ind_melody] = 4
            elif n_new[5]<2:
                new_inst_nos[ind_melody] = 5
        n_new = get_n_new()
        update_t()
        
        if n_new[2]>inst_max and n_new[1]<2:
            ind_chord = t_info.loc[(t_info.new_inst==2)&(t_info.poly_ratio>0)].index
            new_inst_nos[ind_chord] = 1
        n_new = get_n_new()
        update_t()
        
        for _ in range(2):
            if n_new[2]>inst_max:
                empty_inst = np.where(n_new==0)[0]
                empty_inst = empty_inst[empty_inst>0]
                ind_non_poly = t_info.loc[(t_info.new_inst==2)&(t_info.poly_ratio<0.1)].index
                if len(empty_inst)>0 and len(ind_non_poly)>0:
                    new_inst_nos[ind_non_poly[0]] = empty_inst[0]
            n_new = get_n_new()
            update_t()
            
        if n_new[6]>inst_max:
            ind_bass = t_info.loc[(t_info.new_inst==6)&(t_info.mean_note<50)].index
            new_inst_nos[ind_bass] = 3
        n_new = get_n_new()
        update_t()
        
        if n_new[4]>inst_max:
            ind_melody = t_info.loc[(t_info.new_inst==4)&(t_info.mean_note>60)&(t_info.n_note>100)].index
            new_inst_nos[ind_melody[0]] = 5
        n_new = get_n_new()
        update_t()
        
        if n_new[7]>inst_max:
            ind_chord = t_info.loc[(t_info.new_inst==7)&(t_info.poly_ratio>0)].index
            new_inst_nos[ind_chord] = 1
        n_new = get_n_new()
        update_t()
    
    n_new = get_n_new()
    n_t = len(np.where(n_new)[0])
    if n_t<3:
        t_info = pd.DataFrame(describe_tracks(mus))
        def update_t():
            t_info.new_inst = new_inst_nos
        if n_new[1]==2:
            ind_melody = t_info.loc[(t_info.new_inst==1)&(t_info.poly_ratio<0.1)].index
            if len(ind_melody)>0:
                new_inst_nos[ind_melody[0]] = 4
            else:
                ind_mv = t_info.loc[(t_info.new_inst==1)].index[0]
                new_inst_nos[ind_mv] = 2
        n_new = get_n_new()
        update_t()
        
        if n_new[5]==2:
            ind_melody = t_info.loc[(t_info.new_inst==5)&(t_info.poly_ratio<0.1)].index
            new_inst_nos[ind_melody[0]] = 4
        n_new = get_n_new()
        update_t()
        
        if n_new[4]==2:
            ind_melody = t_info.loc[(t_info.new_inst==4)&(t_info.poly_ratio<0.1)].index
            new_inst_nos[ind_melody[0]] = 5
        n_new = get_n_new()
        update_t()
        
        if n_new[7]==2:
            ind_melody = t_info.loc[(t_info.new_inst==7)&(t_info.poly_ratio<0.1)].index
            new_inst_nos[ind_melody[0]] = 4
        n_new = get_n_new()
        update_t()
    return new_inst_nos, n_new


def music_to_proll(mus, return_nz=True):
    """
    Convert music object to pianoroll format and return pianoroll and nonzero index range
    ...
    
    Parameters
    ----------
    mus : muspy.Music
        Music to be converted
    return_nz : bool, optional
        Whether to crop the pianoroll to nonzero content (default True)
        
    Returns
    -------
    
    mus_piano : array
        Pianoroll format music
    nonzero indices : tuple
        First and last location of nonzero indices in converted array before cropping
    """
    mus_piano = muspy.to_pianoroll_representation(mus)
    ind_nz = np.nonzero(mus_piano)
    if return_nz:
        mus_piano = mus_piano[ind_nz[0][0]:ind_nz[0][-1]+1,23:107]
    else:
        mus_piano = mus_piano[:,23:107]
    return mus_piano, (ind_nz[0][0],ind_nz[0][-1]+1)


def mus_to_8track_array(mus,track_pos):
    """
    Convert music object to array of shape (-1,4,96,84,8)
    ...
    
    Parameters
    ----------
    mus : muspy.Music
        Music to be converted
    track_pos : list of int (0-7)
        Position of each mus track in the output array
        
    Returns
    -------
    
    track_rolls : array
        Array of 4 bar samples from mus in 8 tracks
    """
    assert mus.resolution==24
    assert len(track_pos)==len(mus)
    
    full_proll, inds_mus = music_to_proll(mus)
    mus_len = full_proll.shape[0]
    track_rolls = np.zeros((mus_len, 84, 8))
    
    for track, pos in zip(mus,track_pos):
        mus_track = muspy.Music(tracks=[track])
        track_proll, _ = music_to_proll(mus_track,return_nz=False)
        track_proll = track_proll[inds_mus[0]:inds_mus[1],:]
        
        if track_proll.shape[0] < mus_len:
            track_proll = np.pad(track_proll, ((0,mus_len-track_proll.shape[0]), (0,0)) )
        track_rolls[:,:,pos] += track_proll
        
    rem_size = mus_len%(4*96)
    if rem_size != 0:
        track_rolls = track_rolls[:-rem_size,:,:]
    
    return track_rolls.reshape((-1,4,96,84,8))


if __name__ == '__main__':
    """Import all data, determine valid files for converting, and convert valid files"""
    fpath = 'scottish-midi/'
    file_list = os.listdir(fpath)
    all_data = []
    
    print('Loading all midi files...')
    for fname in tqdm(file_list):
        mus_in = muspy.read_midi(fpath+fname)
        all_data.append(describe_data(mus_in))
        
    music_df = pd.DataFrame(data=all_data, columns=['title', 'tracks', 'min_note', 'max_note', 'duration', 'inst_nos', 'inst_names', 'n_drum', 'time_sig', 'tempo'])
    mus_valid = music_df.loc[((music_df.time_sig=='2/4') | (music_df.time_sig=='4/4') | (music_df.time_sig=='2/2'))]
    all_titles = mus_valid.loc[:,'title'].values
    
    all_data_arr = np.empty((0,4,96,84,8))
    print('Converting {} files to arrays'.format(len(mus_valid)))
    for title in tqdm(all_titles):
        fname = title + '.mid'
        mus = muspy.read_midi(fpath+fname)
        mus = mus.adjust_resolution(target=24)
        inst_new, _ = balance_new_tracks(mus)
        mus_arr = mus_to_8track_array(mus,inst_new)

        all_data_arr = np.append(all_data_arr, mus_arr.astype(bool), axis=0)
        
    np.save('conv-scottish-midi-data0',all_data_arr)
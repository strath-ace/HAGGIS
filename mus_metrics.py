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

def props_from_samples(s_path):
    n_file = len(os.listdir(s_path))
    all_f = os.scandir(s_path)
    
    track_inst, track_convert = f_track_converter()
    step_list = []
    dot_list = []
    pent_list = []
    for f in tqdm(all_f, total=n_file):
        
        if not f.name.endswith('round.mid'):
            continue
            
        mus_in = muspy.read_midi(s_path+f.name)
        mus_in = mus_in.adjust_resolution(target=24)
        
        try:
            track_pos = [track_convert[t.program] for t in mus_in]
        except KeyError:
            pdb.set_trace()
        step_list.append(int(f.name[:-15]))
        mus_proll = mus_to_8track_array(mus_in,track_pos)
        
        dot_list.append(dotted_rhythms_8track(mus_proll))
        pent_list.append(pent_notes_8track(mus_proll))
        
        i_sort = np.argsort(step_list)

    return np.array(step_list)[i_sort], np.array(pent_list)[i_sort], np.array(dot_list)[i_sort]

sub_transfer = 5470
if __name__=='__main__':
    s_path = 'samples-test/samples/'
   
    step_list, pent_list, dot_list = props_from_samples(s_path)
    
    fig1 = plt.figure()
    plt.plot(step_list,dot_list)
    plt.title('Dot rhythms')
    plt.xlim((0,np.max(step_list)+500))
    plt.ylim((0,1))
    
    fig2 = plt.figure()
    plt.plot(step_list,pent_list)
    plt.title('Pent notes')
    plt.xlim((0,np.max(step_list)+500))
    plt.ylim((0,1))
    
    plt.show()
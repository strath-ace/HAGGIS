import numpy as np

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

def pent_notes_8track(mus_proll):
    mus_proll = mus_proll.reshape(-1,4,96,84,8)
    mus_tracks = mus_proll[:,:,:,:,1:]

    all_pent_props = []
    all_count_props = []
    for t in tqdm(mus_tracks):
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
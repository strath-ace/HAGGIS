import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import os
import pdb

pentatonic = np.array([1,3,5,8,10])
pent_scales = [pentatonic.copy()]
for _ in range(11):
    pentatonic += 1
    pentatonic[pentatonic>12] -= 12
    pent_scales.append(pentatonic.copy())

print('Loading data...')
their_data = np.load('lastfm_alternative_8b_phrase.npy')[:,:,:,:,:,1:]
their_data = their_data.reshape(-1,4,96,84,7)
print(np.shape(their_data))

all_pent_props = []
all_count_props = []
for p_theirs in tqdm(their_data):
    proll_inds = np.where(p_theirs)
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

print('Mean pentatonic ratio theirs:')
print(np.mean(all_pent_props))
print('Mean pentatonic ratio theirs new:')
print(float(np.sum(np.array(all_pent_props)*np.array(all_count_props)))/float(np.sum(all_count_props)))

counts, bins = np.histogram(all_pent_props,bins=20,range=(0,1))
fig = plt.figure()
h = plt.hist(bins[:-1], bins, weights=counts/np.sum(counts))
x = plt.xlim((0,1))
y = plt.ylim((0,0.3))
plt.title('Theirs')

del(their_data)

print('Loading data...')
our_data = np.load('final-balanced-scottish-midi-data.npy')[:,:,:,:,1:]
print(np.shape(our_data))

all_pent_props = []
all_count_props = []
for p_ours in tqdm(our_data):
    proll_inds = np.where(p_ours)
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

print('Mean pentatonic ratio ours:')
print(np.mean(all_pent_props))
print('Mean pentatonic ratio ours new:')
print(float(np.sum(np.array(all_pent_props)*np.array(all_count_props)))/float(np.sum(all_count_props)))

counts, bins = np.histogram(all_pent_props,bins=20,range=(0,1))
fig = plt.figure()
h = plt.hist(bins[:-1], bins, weights=counts/np.sum(counts))
x = plt.xlim((0,1))
y = plt.ylim((0,0.3))
plt.title('Ours')

plt.show()
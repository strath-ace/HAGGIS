import muspy
import numpy as np


music = muspy.read_midi('../scottish-midi/4marys.mid')
proll = muspy.to_pianoroll_representation(music)
np.savez_compressed("proll_test.npz", shape=proll.shape, nonzero=proll.nonzero())
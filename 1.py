import pickle
with open('/home/share/sunteng/CLUE_model/data.pkl', 'rb') as f:
    data = pickle.load(f)

import numpy as np

key = 'test_nn_text'

raw_text = np.array(data[key]['raw_text'])
audio_lengths = np.array(data[key]['audio_lengths'])
vision_lengths = np.array(data[key]['vision_lengths'])
audio = np.array(data[key]['audio'])
vision = np.array(data[key]['vision'])
regression_labels = np.array(data[key]['regression_labels'])

# %%
with open('{}.pkl'.format(key), 'wb') as f:
  f.write(pickle.dumps({
    'raw_text': raw_text,
    'audio': audio,
    'vision': vision,
    'regression_labels': regression_labels,
    'audio_lengths': audio_lengths,
    'vision_lengths': vision_lengths,
  }))
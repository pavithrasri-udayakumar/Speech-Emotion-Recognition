#importing libraries
import librosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np
#storing audio data into wave format
data, sampling_rate = librosa.load(r'C:\Users\sonia\Downloads\archive (2)\Audio_Speech_Actors_01-24\Actor_01\03-01-05-02-01-02-01.wav')
plt.figure(figsize=(12, 4))
#converting audio to wave
librosa.display.waveshow(data, sr=sampling_rate)
#displaying wave graph
#importing libraries
import time
import os
path = r'C:\Users\sonia\Downloads\archive (2)'
lst = []

start_time = time.time()

for subdir, dirs, files in os.walk(path):
  for file in files:
      try:
        #Load librosa array, obtain mfcss, store the file and the mcss information in a new array
        X, sample_rate = librosa.load(os.path.join(subdir,file), res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
        # The instruction below converts the labels (from 1 to 8) to a series from 0 to 7
        # This is because our predictor needs to start from 0 otherwise it will try to predict also 0.
        file = int(file[7:8]) - 1
        arr = mfccs, file
        lst.append(arr)
      # If the file is not valid, skip it
      except ValueError:
        continue

print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))
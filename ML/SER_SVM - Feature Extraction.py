import numpy as np
import pandas as pd
import time
import librosa
import glob
import os

emotion={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fear',
  '07':'disgust',
  '08':'surprised'
}

tess_emotion=['angry','disgust','fear','ps','happy','sad']
ravdess_emotion=['neutral','calm','angry', 'happy','disgust','sad','fear','surprised']

observed_emotion = ['sad','angry','happy','disgust','surprised','neutral','calm','fear']


def feature_extraction(file_name, mfcc):
  result=None
  X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast')
  if mfcc:
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfccs))
  return result

def dataset_options():
  # choose datasets
  ravdess = True
  tess = True
  ravdess_speech = False
  ravdess_song = False
  data = {'ravdess': ravdess, 'ravdess_speech': ravdess_speech, 'ravdess_song': ravdess_song, 'tess': tess}
  print(data)
  return data


def load_data(test_size=0.3):
  x, y = [], []

  # feature to extract
  mfcc = True

  data = dataset_options()
  paths = []
  if data['ravdess']:
    paths.append("..\features\Actor_*\*.wav")
  elif data['ravdess_speech']:
    paths.append("..\Datasets\RAVDESS\Speech\Actor_*\*.wav")
  elif data['ravdess_song']:
    paths.append("..\Datasets\RAVDESS\Song\Actor_*\*.wav")

  for path in paths:
    for file in glob.glob(path):
      file_name = os.path.basename(file)
      emotion = emotions[
        file_name.split("-")[2]]  # to get emotion according to filename. dictionary emotions is defined above.
      if emotion not in observed_emotion:  # options observed_emotions - RAVDESS and TESS, ravdess_emotions for RAVDESS only
        continue
      feature = feature_extraction(file, mfcc)
      x.append(feature)
      y.append(emotion)
  if data['tess']:
    for file in glob.glob("..\Datasets\TESS\*AF_*.wav"):
      file_name = os.path.basename(file)
      emotion = file_name.split("_")[2][:-4]  # split and remove .wav
      if emotion == 'ps':
        emotion = 'surprised'
      if emotion not in observed_emotion:  # options observed_emotions - RAVDESS and TESS, ravdess_emotions for RAVDESS only
        continue
      feature = feature_extraction(file, mfcc)
      x.append(feature)
      y.append(emotion)
  return {"X": x, "y": y}

start_time = time.time()

Trial_dict = load_data(test_size = 0.3)
print(Trial_dict)

print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))

X = pd.DataFrame(Trial_dict["X"])
y = pd.DataFrame(Trial_dict["y"])

X.shape, y.shape

y=y.rename(columns= {0: 'emotion'})

data = pd.concat([X, y], axis =1)
print(data.head())

data = data.reindex(np.random.permutation(data.index))

data.to_csv("RAVTESS_MFCC_Observed.csv")


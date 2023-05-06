import librosa
import soundfile
import os, glob, pickle
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}
# Emotions to observe
observed_emotions=['neutral','calm','happy','sad','angry','fearful', 'disgust','surprised']


'''def extract_feature(file_name, mfcc, chroma, mel):
  X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast')
  if chroma:
    stft = np.abs(librosa.stft(X))
  result = np.array([])

  if mfcc:
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfccs))
  if chroma:
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma))
  if mel:
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
  return result'''

import librosa
import os
import numpy as np


def extract_feature(file_path, mfcc=True, chroma=True, mel=True):
  with open(file_path, 'rb') as f:
    X, sample_rate = librosa.load(f, sr=None, mono=True)

  features = []
  if mfcc:
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
    features.append(np.mean(mfccs, axis=1))
  if chroma:
    chroma = librosa.feature.chroma_stft(y=X, sr=sample_rate)
    features.append(np.mean(chroma, axis=1))
  if mel:
    mel = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128)
    features.append(np.mean(mel, axis=1))

  return np.concatenate(features)


def load_data(test_size=0.2):
  x, y = [], []
  for file in glob.glob('..\Datasets\RAVDESS\*\Actor_*\*.wav'):
    file_name = os.path.basename(file)
    emotion = emotions[file_name.split("-")[2]]
    if emotion not in observed_emotions:
      continue
    feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
    x.append(feature)
    y.append(emotion)
  return train_test_split(np.array(x), y, test_size=test_size, train_size=0.75, random_state=9)

x_train,x_test,y_train,y_test=load_data(test_size=0.25)

print((x_train.shape[0], x_test.shape[0]))
print(f'Features extracted: {x_train.shape[1]}')
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
model.fit(x_train,y_train)

print("Predict the accuracy of our model")
# Predict for the test set
y_pred=model.predict(x_test)

# Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test,y_pred)
print (matrix)
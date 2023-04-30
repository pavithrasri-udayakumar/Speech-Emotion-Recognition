import librosa
from librosa import display
import matplotlib.pyplot as plt
data, sampling_rate = librosa.load(r'C:\Users\sonia\Downloads\archive (2)\Audio_Speech_Actors_01-24\Actor_01\03-01-05-02-01-02-01.wav')
plt.figure(figsize=(12, 4))
librosa.display.waveshow(data, sr=sampling_rate)
plt.show()
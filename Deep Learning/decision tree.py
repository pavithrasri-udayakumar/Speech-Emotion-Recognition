# library for analysing signals
import librosa
# function for visualising audio signals
from librosa import display
import matplotlib.pyplot as plt
import numpy as np

#loads audio as np array and sampling rate
data, sampling_rate = librosa.load(
    r'C:\Users\sonia\Downloads\archive (2)\Audio_Speech_Actors_01-24\Actor_01\03-01-05-02-01-02-01.wav')
plt.figure(figsize=(12, 4))

# display audio data as wave format
librosa.display.waveshow(data, sr=sampling_rate)
#plt.show()

# importing libraries
import time
import os

path = r'C:\Users\sonia\Downloads\archive (2)\Audio_Speech_Actors_01-24\Actor_06'
lst = []

start_time = time.time()

for subdir, dirs, files in os.walk(path):
    for file in files:
        try:
            # Load librosa array, obtain mfcss, store the file and the mcss information in a new array
            X, sample_rate = librosa.load(os.path.join(subdir, file), res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            # The instruction below converts the labels (from 1 to 8) to a series from 0 to 7
            # This is because our predictor needs to start from 0 otherwise it will try to predict also 0.
            file = int(file[7:8]) - 1
            arr = mfccs, file
            lst.append(arr)
        # If the file is not valid, skip it
        except ValueError:
            continue

print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))
X, y = zip(*lst)
X = np.asarray(X)
y = np.asarray(y)
print(X.shape)
print(y.shape)

import joblib
X_name = 'X.joblib'
y_name = 'y.joblib'
save_dir = r'C:\Users\sonia\Downloads\archive (2)'

savedX = joblib.dump(X, os.path.join(save_dir, X_name))
savedy = joblib.dump(y, os.path.join(save_dir, y_name))

import joblib
X = joblib.load(r'C:\Users\sonia\Downloads\archive (2)\X.joblib')
y = joblib.load(r'C:\Users\sonia\Downloads\archive (2)\y.joblib')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions,zero_division=1))
x_traincnn = np.expand_dims(X_train, axis=2)
x_testcnn = np.expand_dims(X_test, axis=2)
print(x_traincnn.shape, x_testcnn.shape)
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential



model = Sequential()

model.add(Conv1D(64, 5,padding='same',
                 input_shape=(40,1)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(4)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(4)))
model.add(Conv1D(256, 5,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(8))
model.add(Activation('softmax'))
opt = RMSprop(learning_rate=0.00005, rho=0.9, epsilon=1e-07, decay=0.0)
model.summary()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=200, validation_data=(x_testcnn, y_test))

plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(cnnhistory.history['accuracy'])
plt.plot(cnnhistory.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

predictions = np.argmax(model.predict(x_testcnn), axis=-1)

predictions

y_test
new_Ytest = y_test.astype(int)
new_Ytest

from sklearn.metrics import classification_report
report = classification_report(new_Ytest, predictions)
print(report)
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(new_Ytest, predictions)
print (matrix)

# 0 = neutral, 1 = calm, 2 = happy, 3 = sad, 4 = angry, 5 = fearful, 6 = disgust, 7 = surprised

model.save('testing10_model.h5')
print("MODEL SAVED")

new_model=keras.models.load_model('testing10_model.h5')
new_model.summary()

loss, acc = new_model.evaluate(x_testcnn, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))







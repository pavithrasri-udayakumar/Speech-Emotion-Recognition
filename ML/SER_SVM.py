import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
data = pd.read_csv('RAVTESS_MFCC_Observed.csv')


starting_time = time.time()
data= pd.read_csv('..\ML\RAVTESS_MFCC_Observed.csv')
#data = pd.read_csv('/content/drive/My Drive/SER/Dataset/RAVTESS_MFCC_Observed.csv')
data.isna().sum()
print("data loaded in " + str(time.time()-starting_time) + "ms")


data = data.drop('Unnamed: 0',axis=1)

#df = data.drop('0', axis=1)
print(data.columns)
print(data.head())


X = data.drop('emotion', axis = 1).values
y = data['emotion'].values



print(X.shape, y.shape)
np.unique(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
print(X_train,y_train)


svclassifier = SVC(kernel = 'linear')



starting_time = time.time()
svclassifier.fit(X_train, y_train)
print("Trained model in %s ms " % str(time.time() - starting_time))



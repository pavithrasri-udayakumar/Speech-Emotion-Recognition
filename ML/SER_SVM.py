import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
data = pd.read_csv('RAVTESS_MFCC_Observed.csv')


starting_time = time.time()
data= pd.read_csv('..\ML\RAVTESS_MFCC_Observed.csv')
data.isna().sum()
print("data loaded in " + str(time.time()-starting_time) + "ms")


data = data.drop('Unnamed: 0',axis=1)

data= data.drop('0', axis=1)
print(data.columns)
print(data.head())


X = data.drop('emotion', axis = 1).values
y = data['emotion'].values



print(X.shape, y.shape)
np.unique(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


svclassifier = SVC(kernel = 'linear')



starting_time = time.time()
svclassifier.fit(X_train, y_train)
print("Trained model in %s ms " % str(time.time() - starting_time))

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import seaborn as sn

print(classification_report(y_test,y_pred))

acc = float(accuracy_score(y_test,y_pred))*100
print("----accuracy score %s ----" % acc)

cm = confusion_matrix(y_test,y_pred)
df_cm = pd.DataFrame(cm)
sn.heatmap(df_cm, annot=True, fmt='')
plt.show()

train_acc = float(svclassifier.score(X_train, y_train)*100)
print("----train accuracy score %s ----" % train_acc)

test_acc = float(svclassifier.score(X_test, y_test)*100)
print("----test accuracy score %s ----" % test_acc)

#cross-validation
cv_results = cross_val_score(svclassifier, X, y, cv = 5)
print(cv_results)






#Audio based Speech-Emotion-Recognition using AI techniques
SONIA FRANCIS JAVIOR:
#Install the required libraries 
-keras: neural network library to deal with deep learning models
-librosa:processing the audio files to wave format
-numpy:working with numerical data
-tensorflow: building deep learning models
-scikit-learn: for preprocessing data, feature extraction and model evaluation
-matplotlib:for visualising interactive data

Step1: COllection of Dataset
•	Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) dataset
•	Toronto emotional speech set (TESS) dataset
Step2:
Go to tess.py --> run the code to combine the dataset of RAVDESS AND TESS.
              -->this will be done using pipelines and create .joblib extension files to perform parallel processing in audio signal.
Step3:
Go to cnn.py --> run the code
O/P:
You will the loss against test data, accuracy of test data, overall score of test data.The pretained model is stored as ".h5" extension.
the pretrained model is evaluated under live data and predicted with overall accuracy of 65%.

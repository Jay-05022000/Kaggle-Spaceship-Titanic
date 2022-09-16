# Importing libraries.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix

# Creating a Dataframe.

train_dataset=pd.DataFrame(pd.read_excel('Cleaned_train.xlsx'))
test_features=pd.DataFrame(pd.read_excel('Cleaned_test.xlsx'))
Submission=pd.read_csv('sample_submission.csv')

features_train=train_dataset.iloc[:,:-1].values
label_train=train_dataset.iloc[:,-1]


# feature scaling.

# scaling whole dataset.

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sct=StandardScaler()
features_train=sc.fit_transform(features_train )
test_features=sct.fit_transform(test_features)


# Splitting test dataset for model training & calculating test accuracy.
''' 
from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test=train_test_split(features_train,label_train,test_size=0.107,random_state=0)   # test count=930.04

from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(x_train,y_train)

predicted_y_test=classifier.predict(x_test)

cm=confusion_matrix(y_test,predicted_y_test)
print(cm)
print(accuracy_score(y_test,predicted_y_test))

'''
# SVC model training.
classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(features_train,label_train )
predicted_label_test=pd.Series(classifier.predict(test_features))

# Creating a submission file
Submission.Transported=predicted_label_test.astype('bool')
print(Submission)
Submission.to_csv('result(2).csv',index=False)

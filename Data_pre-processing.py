# Importing libraries.

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC

# creating a Dataframe.
df1=pd.DataFrame(pd.read_excel('train_excel.xlsx'))


# --------------------------Data-preprocessing------------------------------------

# Encoding categorical values using LabelEncoder.

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

N_P_ID=le.fit_transform(df1['PassengerId'])
N_HP=le.fit_transform(df1['HomePlanet'])
N_Cs=le.fit_transform(df1['CryoSleep'])
N_Cb=le.fit_transform(df1['Cabin'])
N_Ds=le.fit_transform(df1['Destination'])
N_Vip=le.fit_transform(df1['VIP'])
N_Tp=le.fit_transform(df1['Transported'])


df1.drop('PassengerId',axis=1,inplace=True)
df1.drop('HomePlanet' ,axis=1,inplace=True)
df1.drop('CryoSleep',axis=1,inplace=True)
df1.drop('Cabin' ,axis=1,inplace=True)
df1.drop('Destination' ,axis=1,inplace=True)
df1.drop('VIP',axis=1,inplace=True)
df1.drop('Transported' ,axis=1,inplace=True)

df1['PassengerId']=N_P_ID
df1['HomePlanet']=N_HP
df1['CryoSleep']=N_Cs
df1['Cabin']=N_Cb
df1['Destination']=N_Ds
df1['VIP']=N_Vip
df1['Transported']=N_Tp

# filling null values using KNNImputer function.

from sklearn.impute import KNNImputer

Imputer=KNNImputer(n_neighbors=3)
New_df1=Imputer.fit_transform(df1)

Cleaned_df=pd.DataFrame(New_df1,columns=['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck', 'PassengerId','HomePlanet','CryoSleep','Cabin','Destination','VIP','Transported' ])
print(Cleaned_df.head(10))
Cleaned_df.to_excel('Cleaned_train(1).xlsx')

# Feature selction using correlation coefficient matrix.

correlation= Cleaned_df.corr()
plt.figure(figsize=(10,6))
x=sns.heatmap(correlation.round(3),annot=True)
plt.show()



import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data  = pd.read_csv('C:/data/csv/winequality-white.csv',sep=';')
print(data['quality'].value_counts())   
print(data.describe())

wine_npy = data.values

x_data = wine_npy[:,:11]
y_data = wine_npy[:,11]

from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,QuantileTransformer  #이상치 제거를 하지 않은 상태에 RobustSCALER 사용하면 standardscaler보다 효과적이다 

x_train,x_test, y_train, y_test = train_test_split(x_data,y_data, train_size =0.8, shuffle=True, random_state=66)

scale = StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

model = XGBClassifier()
model.fit(x_train, y_train)
score = model.score(x_test,y_test)
print("score : ",score)
# KNeighborsClassifier score :  0.5663265306122449
# RandomForestClassifier score :  0.710204081632653
# XGBClassifier score :  0.6816326530612244





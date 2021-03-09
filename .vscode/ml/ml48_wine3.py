import pandas as pd
import warnings
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  
import seaborn as sns
import matplotlib.pyplot as plt

wine  = pd.read_csv('C:/data/csv/winequality-white.csv',sep=';')
print(wine['quality'].value_counts())   
print(wine.describe())

x_data = wine.iloc[:,0:-1]
y_data = wine.iloc[:,-1]

newlist=[]
for i in list(y_data):
    if i <=4:
        newlist +=[0]
    elif i <=7:
        newlist +=[1]
    else:
        newlist +=[2]

y_data = newlist 

from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x_data,y_data, train_size =0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import PowerTransformer,MinMaxScaler,StandardScaler,RobustScaler,QuantileTransformer  #이상치 제거를 하지 않은 상태에 RobustSCALER 사용하면 standardscaler보다 효과적이다 

#디폴트 : 균등분포
scaler = PowerTransformer(method='yeo-johnson')
scaler.fit(x_train)
x_train=scaler.transform(x_train) 

from sklearn.metrics import classification
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression()
model.fit(x_train,y_train)

y_true, y_pred = y_test, model.predict(x_test)
print(classification_report(y_true,y_pred))
result = model.score(x_test,y_test)
print('model_score : ', result)
accuracy = accuracy_score(y_pred,y_test)
print('accuracy_score : ', accuracy)

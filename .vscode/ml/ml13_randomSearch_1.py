import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris   #다중 분류모델
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split,KFold,cross_val_score, GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  #회기가 아니라 분류이다
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
dataset = load_iris()

x = dataset.data
y= dataset.target



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=66)
kfold = KFold(n_splits=5, shuffle=True) 
parameters = [
    {"C":[1,10,100,1000],"kernel":["linear"]},
    {"C":[1,10,100],"kernel":["rbf"],"gamma":[0.001,0.0001]},
    {"C":[1,10,100,1000],"kernel":["sigmoid"],"gamma":[0.001,0.0001]}
] #4+6+8 총 18번 돈다 cross_val 총 5번이기때문에 18x5 90번돈다
#model = SVC()
#2.model
model = RandomizedSearchCV(SVC(),parameters,cv=kfold)  #SVC모델을 그리드써치로 싸버리겠다는 

#3.compile,trainning
model.fit(x_train,y_train)

#4.evaluate,predict
print("최적의 매개변수:",model.best_estimator_)
y_pred = model.predict(x_test)
print('최종정답률:',accuracy_score(y_test,y_pred))

#최적의 매개변수: SVC(C=1000, gamma=0.001, kernel='sigmoid')
#최종정답률: 0.9666666666666667


#파라미터를 다 안돌리고 랜덤으로 돌리기때문에 그리드서치보다 훨씬 빠름

#randomizedsearch에 적용되는 랜덤의 퍼센트가 몇퍼센트인지 찾아내기!
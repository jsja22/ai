import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris   #다중 분류모델
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  #회기가 아니라 분류이다
import matplotlib.pyplot as plt
dataset = load_iris()

x = dataset.data
y= dataset.target

kfold = KFold(n_splits=5, shuffle=True)

model =LinearSVC()
score = cross_val_score(model,x,y,cv=kfold)  #model and data concatenate

print('score :',score) #fit하고 model.score까지 포함
'''
model.fit(x_train,y_train)

result = model.score(x_test,y_test)
print(result) #(loss,acuracy)
 
print("test predict: {}".format(model.predict(x_test))) 
print("test accuracy: {:.2f}".format(result))  
#test predict: [1 1 1 0 1 1 0 0 0 1 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 1 2]
#test accuracy: 0.93


#acc total
##KNeighbors :0.97
##DecisionTRee : 0.93
#RandomForest : 0.93

#test predict: [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 2 2]
#test accuracy: 1.00

#model.score가 model. evaluate보다 먼저였음
#sklearn에서 model.score과 keras에서 acc는 같음

##iris 이진 분류모델에서는 LogisticRegression 가 성능이 젤 좋았다

#tensorflow
#Dense model acc-> 1
#LSTM model -> acc: 0.9667

#LSTM 보단 머신러닝 acc가 높게 나왔다
'''
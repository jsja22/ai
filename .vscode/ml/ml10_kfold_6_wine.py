#6개의 모델 완성

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris ,load_wine  #다중 분류모델
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  #회기가 아니라 분류이다
import matplotlib.pyplot as plt
dataset = load_wine()

x = dataset.data
y= dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=77, shuffle =True, train_size = 0.8)
scalar = MinMaxScaler()
scalar.fit(x_train)
x_train=scalar.transform(x_train)
x_test = scalar.transform(x_test)

kfold = KFold(n_splits=5, shuffle=True)


models = [LinearSVC(), SVC(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), LogisticRegression()]
for i in models:
    model = i
    score = cross_val_score(model,x_train,y_train,cv=kfold)  #model and data concatenate
    #3. compile fit
    model.fit(x_train,y_train)

    #4. evaluation, prediction
    print(f'\n{i}')
    result = model.score(x_test,y_test)
    print('model_score : ', result)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_pred,y_test)
    print('accuracy_score : ', accuracy)
    print('score :',score)

    
####Before Minmax ####################
# LinearSVC()
# model_score :  0.8888888888888888
# accuracy_score :  0.8888888888888888
# score : [0.75862069 0.86206897 0.53571429 0.96428571 0.75      ]

# SVC()
# model_score :  0.6388888888888888
# accuracy_score :  0.6388888888888888
# score : [0.55172414 0.5862069  0.71428571 0.78571429 0.67857143]

# KNeighborsClassifier()
# model_score :  0.7222222222222222
# accuracy_score :  0.7222222222222222
# score : [0.72413793 0.62068966 0.71428571 0.67857143 0.5       ]

# DecisionTreeClassifier()
# model_score :  0.9166666666666666
# accuracy_score :  0.9166666666666666
# score : [0.96551724 0.82758621 0.92857143 0.82142857 0.89285714]

# RandomForestClassifier()
# model_score :  1.0
# accuracy_score :  1.0
# score : [1.         1.         0.85714286 1.         1.        ]



# LogisticRegression()
# model_score :  0.9166666666666666
# accuracy_score :  0.9166666666666666
# score : [0.93103448 0.93103448 0.92857143 1.         0.96428571]



####After Minmax ####################

# LinearSVC()
# model_score :  1.0
# accuracy_score :  1.0
# score : [0.96551724 0.96551724 0.96428571 1.         0.96428571]

# SVC()
# model_score :  1.0
# accuracy_score :  1.0
# score : [1.         0.96551724 0.96428571 0.96428571 0.96428571]

# KNeighborsClassifier()
# model_score :  1.0
# accuracy_score :  1.0
# score : [0.93103448 1.         0.89285714 0.96428571 1.        ]

# DecisionTreeClassifier()
# model_score :  0.8888888888888888
# accuracy_score :  0.8888888888888888
# score : [0.86206897 0.82758621 0.71428571 0.92857143 0.96428571]

# RandomForestClassifier()
# model_score :  1.0
# accuracy_score :  1.0
# score : [1.         1.         0.92857143 1.         0.92857143]

# LogisticRegression()
# model_score :  1.0
# accuracy_score :  1.0
# score : [0.93103448 1.         1.         0.96428571 1.        ]
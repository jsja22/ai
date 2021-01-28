#6개의 모델 완성

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

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=77, shuffle =True, train_size = 0.8)


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




# LinearSVC()
# model_score :  0.9
# accuracy_score :  0.9
# score : [0.91666667 1.         1.         0.91666667 1.        ]

# SVC()
# model_score :  0.9
# accuracy_score :  0.9
# score : [0.95833333 0.91666667 0.95833333 1.         1.        ]

# KNeighborsClassifier()
# model_score :  0.9333333333333333
# accuracy_score :  0.9333333333333333
# score : [1.         0.95833333 1.         1.         1.        ]

# DecisionTreeClassifier()
# model_score :  0.8666666666666667
# accuracy_score :  0.8666666666666667
# score : [1.         1.         0.83333333 0.95833333 0.95833333]

# LogisticRegression()
# model_score :  0.8666666666666667
# accuracy_score :  0.8666666666666667
# score : [0.95833333 0.95833333 1.         0.95833333 0.91666667]
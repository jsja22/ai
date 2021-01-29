#25 % 미만인 컬럼들을 제거

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris,load_breast_cancer,load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def feature_importances_find(model):
    f_importance = model.feature_importances_
    f_columns = sorted(range(len(f_importance)),key=lambda i:f_importance[i],reverse=True)[-1*round(len(f_importance)*0.25):]

    return f_columns


dataset = load_wine()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data,dataset.target,train_size=0.8,random_state=44)

print(x_train.shape)
print(x_test.shape)

model1 = RandomForestClassifier()
model1.fit(x_train,y_train)
f_columns = feature_importances_find(model1)

x_train = np.delete(x_train, f_columns, axis=1)
x_test = np.delete(x_test, f_columns, axis=1)

print(x_train.shape)
print(x_test.shape)

model2 = RandomForestClassifier()
model2.fit(x_train,y_train)

acc = model2.score(x_test,y_test)

print(model2.feature_importances_)
print("acc:",acc)

# [0.15645165 0.02114674 0.01161654 0.04482256 0.17615118 0.02146342
#  0.15810593 0.06272446 0.14558814 0.20192939]
# acc: 0.9722222222222222
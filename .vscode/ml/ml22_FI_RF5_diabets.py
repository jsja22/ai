#25 % 미만인 컬럼들을 제거

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris,load_breast_cancer,load_boston,load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def feature_importances_find(model):
    f_importance = model.feature_importances_
    f_columns = sorted(range(len(f_importance)),key=lambda i:f_importance[i],reverse=True)[-1*round(len(f_importance)*0.25):]

    return f_columns


dataset = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data,dataset.target,train_size=0.8,random_state=44)

print(x_train.shape)
print(x_test.shape)

model1 = RandomForestRegressor()
model1.fit(x_train,y_train)
f_columns = feature_importances_find(model1)

x_train = np.delete(x_train, f_columns, axis=1)
x_test = np.delete(x_test, f_columns, axis=1)

print(x_train.shape)
print(x_test.shape)

model2 = RandomForestRegressor()
model2.fit(x_train,y_train)

acc = model2.score(x_test,y_test)

print(model2.feature_importances_)
print("acc:",acc)

# [0.07433622 0.24912398 0.08054522 0.04779922 0.06693255 0.05853178
#  0.34454762 0.07818341]
# acc: 0.40435720990778057
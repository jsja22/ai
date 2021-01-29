#25 % 미만인 컬럼들을 제거

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def feature_importances_find(model):
    f_importance = model.feature_importances_
    f_columns = sorted(range(len(f_importance)),key=lambda i:f_importance[i],reverse=True)[-1*round(len(f_importance)*0.25):]

    return f_columns


dataset = load_breast_cancer()
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

# [0.02806694 0.01983102 0.06433383 0.03685465 0.00869103 0.01455561
#  0.0451493  0.11751282 0.00455142 0.02195966 0.00488278 0.03583908
#  0.12013103 0.02113972 0.13311901 0.1286671  0.0202835  0.01912439
#  0.02453114 0.10973879 0.012232   0.00880517]
# acc: 0.9649122807017544
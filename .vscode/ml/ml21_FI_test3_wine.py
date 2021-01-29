#피처임포턴스가 0인 컬럼들을 제거하여 데이터셋을 재구성후 Desisiontree 모델을 재구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris,load_breast_cancer,load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = load_wine()
print(dataset.feature_names)
x_train,x_test,y_train,y_test = train_test_split(dataset.data,dataset.target,train_size = 0.8, random_state = 33)
model1 = DecisionTreeClassifier(max_depth = 5)
model1.fit(x_train, y_train)
acc = model1.score(x_test,y_test)

print(model1.feature_importances_)
print('acc before delete columns: ', acc)

f_importance = model1.feature_importances_
f_data = []
f_columns =[]
for i in range(len(f_importance)):
    if f_importance[i] !=0:
        f_data.append(dataset.data[:,i])
        f_columns.append(dataset.feature_names[i])

f_data = np.array(f_data)
f_data = np.transpose(f_data)

x_train2, x_test2, y_train2, y_test2 = train_test_split(
    f_data,dataset.target,train_size=0.8,random_state=44)


model2 = DecisionTreeClassifier(max_depth=4)
model2.fit(x_train2,y_train2)

acc2 = model2.score(x_test2,y_test2)

print(model2.feature_importances_)
print("acc delete 0 columns:",acc2)

def plot_feature_importances_dataset(model):
    n_features = f_data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),f_columns)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)

plot_feature_importances_dataset(model2)
plt.show().
# [0.01980069 0.         0.         0.         0.         0.
#  0.06066412 0.         0.         0.0605901  0.07278431 0.32130798
#  0.4648528 ]
# acc before delete columns:  0.7777777777777778
# [0.         0.17679511 0.0179565  0.05577403 0.32933594 0.42013842]
# acc delete 0 columns: 0.9166666666666666
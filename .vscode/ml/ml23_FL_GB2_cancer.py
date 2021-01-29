#25 % 미만인 컬럼들을 제거

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
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

model1 = GradientBoostingClassifier()
model1.fit(x_train,y_train)
def plot_feature_importances_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)

plot_feature_importances_dataset(model1)
plt.show()

f_columns = feature_importances_find(model1)

x_train = np.delete(x_train, f_columns, axis=1)
x_test = np.delete(x_test, f_columns, axis=1)

print(x_train.shape)
print(x_test.shape)

model2 = GradientBoostingClassifier()
model2.fit(x_train,y_train)

acc = model2.score(x_test,y_test)

print(model2.feature_importances_)
print("acc:",acc)

# [0.01037958 0.00148345 0.00301772 0.00067894 0.41135415 0.00175511
#  0.00297338 0.00275733 0.00216499 0.00841265 0.00138173 0.00103045
#  0.00088295 0.00195355 0.07098622 0.06092121 0.2842926  0.04475637
#  0.01361277 0.00232819 0.01920061 0.05367606]
# acc: 0.9649122807017544
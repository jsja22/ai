import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris   #다중 분류모델
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
dataset = load_iris()

x = dataset.data
y= dataset.target
# from sklearn.preprocessing import MinMaxScaler #x값을 minmax전처리 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=66)


model = DecisionTreeClassifier(max_depth=7, random_state=1)

from tensorflow.keras.callbacks import EarlyStopping

model.fit(x_train,y_train)

result = model.score(x_test,y_test)
print(result) #(loss,acuracy)
 
print("test predict: {}".format(model.predict(x_test))) 
print("test accuracy: {:.2f}".format(result))  
#test predict: [1 1 1 0 1 1 0 0 0 1 2 2 0 1 2 0 1 1 2 2 0 1 1 2 1 2 0 0 2 2]
#test accuracy: 0.93
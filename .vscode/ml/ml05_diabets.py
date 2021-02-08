import numpy as np
import tensorflow as tf
from sklearn.datasets import load_diabetes  #다중 분류모델
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score  #회귀모델에서는 r2_score로 평가해야함 #분류에서는 accuracy_score이다
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

dataset = load_diabetes()
x= dataset.data
y= dataset.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True  ,random_state=66)
#x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, train_size=0.2,shuffle=True,random_state=66)

scalar = StandardScaler()
scalar.fit(x_train)
x_train=scalar.transform(x_train)
x_test = scalar.transform(x_test)
#x_val = scalar.transform(x_val)

model =DecisionTreeRegressor()

model.fit(x_train,y_train)

result = model.score(x_test,y_test)
print(result) #(loss,acuracy)

y_predict = model.predict(x_test)
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
#print("mse : ", mean_squared_error(y_test, y_predict))


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


########회귀모델에서는 r2_score로 평가해야함 #분류에서는 accuracy_score이다$#####
#############MinmaxScaler###########
# KNeighborsClassifier(n_neighbors =3)
#0.0
#RMSE :  92.23108496438641
#R2 :  -0.3107118770936834


#DecisionTreeRegressor
#-0.20172552273875421
#RMSE :  88.3133466187939
#R2 :  -0.20172552273875421

#RandomForestClassifier
#0.36576571837904914
#RMSE :  64.15766602762572
#R2 :  0.36576571837904914


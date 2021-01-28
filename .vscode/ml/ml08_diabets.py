import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris ,load_breast_cancer,load_boston,load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
dataset = load_diabetes()

x = dataset.data
y= dataset.target
print(dataset.feature_names)
print(dataset.DESCR)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=66)

scalar =StandardScaler()
scalar.fit(x_train)
x_train=scalar.transform(x_train)
x_test = scalar.transform(x_test)

print(x_test.shape) #(89,10)
models = [LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor()]
for i in models:
    model = i

    #3. compile fit
    model.fit(x_train,y_train)

    #4. evaluation, prediction
    print(f'\n{i}')
    result = model.score(x_test,y_test)
    print('model_score : ', result)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test,y_pred)
    print('r2_score : ', r2)
###Minmax
#LinearRegression()
#model_score :  0.5063891053505036
#r2_score :  0.5063891053505036

#KNeighborsRegressor()
#model_score :  0.3741821819765594
#r2_score :  0.3741821819765594

#DecisionTreeRegressor()
#model_score :  -0.17987008149182593
#r2_score :  -0.17987008149182593

#RandomForestRegressor()
#model_score :  0.37380544794243487
#r2_score :  0.37380544794243487

#성능은 LinearRegression가 가장 우수

##Standard
# LinearRegression()
# model_score :  0.5063891053505039
# r2_score :  0.5063891053505039

# KNeighborsRegressor()
# model_score :  0.38626977834604637
# r2_score :  0.38626977834604637

# DecisionTreeRegressor()
# model_score :  -0.22370215226889578
# r2_score :  -0.22370215226889578

# RandomForestRegressor()
# model_score :  0.35315539712505595
# r2_score :  0.35315539712505595
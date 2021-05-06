import numpy as np
import tensorflow as tf

x= np.array([range(100),range(301,401),range(1,101)])    
y= np.array([range(711,811),range(1,101),range(100)])


print(x.shape)  #(3,100)
print(y.shape)  #(100, )

x= np.transpose(x)
y= np.transpose(y)

print(x)
print(x.shape)

from sklearn.model_selection import train_test_split  #행을 자르겠다는 뜻
x_train,x_test,y_train,y_test = train_test_split(x,y, shuffle=True, train_size=0.8,random_state=66)

print(x_train.shape)    #(80,3)
print(y_train.shape)    #(80,3)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense
#
model = Sequential()
model.add(Dense(20,input_dim = 3,activation='relu'))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(3))


model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100, batch_size=1, validation_split=0.2)


loss,mae = model.evaluate(x_test,y_test)
print('loss:',loss)
print('mae:',mae)

y_predict = model.predict(x_test)
#print(y_predict)


from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ",RMSE(y_test,y_predict))

from sklearn.metrics import r2_score 
r2 = r2_score(y_test,y_predict)
print("R2 :",r2)

import numpy as np
import tensorflow as tf

x= np.array([range(100),range(301,401),range(1,101)])    
y= np.array(range(711,811))

#
#print(x.shape)  #(3,100)
#print(y.shape)  #(100, )
#x= np.arange(20).reshape(10,2)

x= np.transpose(x)
#print(x)
'''
x_train = x[:70]
x_test = x[70:100]

y_train = y[:70]
y_test = y[70:100]
#print(x_train.shape)  (100, 3)
'''
from sklearn.model_selection import train_test_split  #행을 자르겠다는 뜻
x_train,x_test,y_train,y_test = train_test_split(x,y, shuffle=True, train_size=0.8,random_state=66)
print(x_train.shape)
print(y_train.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense

model = Sequential()
model.add(Dense(20,input_dim = 3,activation='relu'))
model.add(Dense(256))
model.add(Dense(512))
model.add(Dense(512))
model.add(Dense(216))
model.add(Dense(64))
model.add(Dense(1))


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

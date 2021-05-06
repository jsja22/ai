import numpy as np
x= np.array([[1,2,3,4,5,6,7,8,9,10],
            [11,12,13,14,15,16,17,18,19,20]])
y= np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape)
#x= np.arange(20).reshape(10,2)

x= np.transpose(x)
print(x)
print(x.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense

model = Sequential()
model.add(Dense(20,input_dim = 2,activation='relu'))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(5))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x,y,epochs=100, batch_size=1, validation_split=0.2)

#
loss,mae = model.evaluate(x,y)
print('loss:',loss)
print('mae:',mae)

y_predict = model.predict(x)
#print(y_predict)

'''
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
#print("RMSE : ",RMSE(y_test,y_predict))
#print("mse :",mean_squared_error(y_predict,y_test))

from sklearn.metrics import r2_score 
r2 = r2_score(y_test,y_predict)
print("R2 :",r2)
'''
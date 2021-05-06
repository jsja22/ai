#실습2
#1. r2 : 0.5이하/음수는 안댐
#2. layer : 5개이상
#3. node : 각 10개 이상
#4.batch_size : 8이하
#5. epochs : 30이상
##
import numpy as np
import tensorflow as tf

x= np.array([range(100),range(301,401),range(1,101),range(100),range(301,401)])    
y= np.array([range(711,811),range(1,101)])


print(x.shape)  
print(y.shape)  

x= np.transpose(x)
y= np.transpose(y)

#print(x)
print(x.shape)
print(y.shape)
x_pred2 = np.array([100,402,101,100,410])
print("x_pred2.shape :", x_pred2.shape)    #(5, )
x_pred2 = np.transpose(x_pred2)
print("x_pred2.shape :", x_pred2.shape)    #(5, )     => 즉 1차원적인 스칼라 집합이기에 행렬이 아니라서 전치가 안된다. 따라서 x_pre2 = x_pred2.reshape(1, 5) 로 해줘야함
x_pred2 = x_pred2.reshape(1, 5)   # => 2차원행렬로 바뀌었음. 
print("x_pred2.shape :", x_pred2.shape)    



from sklearn.model_selection import train_test_split  #행을 자르겠다는 뜻
x_train,x_test,y_train,y_test = train_test_split(x,y, shuffle=True, train_size=0.8,random_state=66)

print(x_test)

print(x_train.shape)    #(80,3)
print(y_train.shape)    #(80,3)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense

model = Sequential()
model.add(Dense(50,input_dim = 5,activation='relu'))
model.add(Dense(256))
model.add(Dense(30000))
model.add(Dense(256))
model.add(Dense(64))
model.add(Dense(2))


model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=30, batch_size=8, validation_split=0.2)    


loss,mae = model.evaluate(x_test,y_test)    # x_test 20:5 / y_test 20:2
print('loss:',loss)
print('mae:',mae)

y_predict = model.predict(x_test)   #y_test와 유사한값이 나옴 
#print(y_predict)


from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ",RMSE(y_test,y_predict))

from sklearn.metrics import r2_score 
r2 = r2_score(y_test,y_predict)
print("R2 :",r2)


y_pred2 = model.predict(x_pred2)
print(y_pred2)



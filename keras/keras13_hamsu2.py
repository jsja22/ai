import numpy as np
import tensorflow as tf

x= np.array([range(100),range(301,401),range(1,101),range(100),range(301,401)])    
y= np.array([range(711,811),range(1,101)])



print(x.shape)  
print(y.shape)  

x= np.transpose(x)
y= np.transpose(y)
#
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
x_train,x_test,y_train,y_test = train_test_split(x,y, shuffle=True, train_size=0.8,random_state=66)  #random 난수표에 맞춰서 돌아간다. 66번 위치에 맞춰 돌리는것이기에 다음 데이터도 그 위치에 맞게 돌아간다.
                                                                                                     #즉 랜덤 난수에 66번째 위치를 사용하겠다는 의미   
print(x_test)

print(x_train.shape)    #(80,3)
print(y_train.shape)    #(80,3)

from tensorflow.keras.models import Sequential, Model   #함수형 모델
from tensorflow.keras.layers import Dense, Input
# from keras.layers import Dense

input1 = Input(shape=(5,))   #INPUT LAYER를 직접 구성하겠다는 뜻
aaa = Dense(5, activation='relu')(input1)
aaa = Dense(3)(aaa)
aaa = Dense(4)(aaa)
outputs = Dense(2)(aaa)    #aaa로 해도 알아서 업데이트를 해서 구동된다. 
model = Model(inputs= input1,outputs = outputs)
model.summary()    


#=> model.add(Dense(10,activation='relu',input_shape = (5,))) 이거와 같음
#=> 함수형 모델
'''
model = Sequential()
model.add(Dense(5, activation='relu',input_shape = (1,)))
model.add(Dense(3))  #동일함 이렇게도 표현가능
model.add(Dense(4))
model.add(Dense(1))
model.summary()
'''
#=> seguential 모델     함수형과 시퀀스 성능은 동일하다. 

model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100, batch_size=1, validation_split=0.2,verbose=0)    #verbose=0이면 훈련되는 과정을 보지 않게 한다. (시간적으로 장점이 있음->훈련이 빨라짐, 대신에 과정은 볼수없다는 단점)


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



import numpy as np

#1.데이터
x= np.array([1,2,3,4,5,6,7,8,9,10])
y= np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2.모델구성
model = Sequential()
model.add(Dense(1000,input_dim=1))
model. add(Dense(100))
model. add(Dense(100))
model. add(Dense(100))
model. add(Dense(100))
model. add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop,SGD, Nadam

optimizer = Nadam(lr=0.001)

model.compile(loss='mse',optimizer=optimizer, metrics=['mse'])
model.fit(x,y,epochs=100, batch_size = 1)

#4.평가 예측
loss, mse = model.evaluate(x, y, batch_size= 1)
y_pred = model.predict([11])
print("loss:", loss, "결과물: ", y_pred)


################Adam###########################(lr=0.01)
#lr=0.001
#loss: 4.435207843350719e-12 결과물:  [[10.999998]]
#lr = 0.01
#loss: 2.9558576723417995e-13 결과물:  [[11.]]
#lr = 0.1
#loss: 0.0005328037077561021 결과물:  [[11.101657]]

################Adadelta###########################(lr=0.01)
#lr = 0.1
#loss: 0.012485286220908165 결과물:  [[10.790512]]
#lr = 0.01
#loss: 6.691928865620866e-05 결과물:  [[10.982355]]
#lr = 0.001
#5.331675052642822 결과물:  [[6.834516]]

################Adamax###########################(lr=0.01)
#lr = 0.1
#3.0197503519957536e-07 결과물:  [[10.999675]]
#lr = 0.01
#1.5128875582859358e-11 결과물:  [[11.000004]]
#lr = 0.001
#loss: 5.369753353079432e-07 결과물:  [[11.000524]]

################Adagrad###########################(lr=0.001)
#lr = 0.1
#225.796630859375 결과물:  [[-21.356308]]
#lr = 0.01
#loss: 4.945795808453113e-06 결과물:  [[11.000961]]
#lr = 0.001
#loss: 3.2747211662353948e-06 결과물:  [[10.9971]]

################RMSprop###########################(lr=0.01)
#lr = 0.1
#loss: 171926306816.0 결과물:  [[-856648.06]]
#lr = 0.01
#loss: 0.34856539964675903 결과물:  [[11.186613]]
#lr = 0.001
#loss: 0.4838119447231293 결과물:  [[9.636472]]

################SGD###########################(lr=0.001)
#lr = 0.1
#loss: nan 결과물:  [[nan]]  #자기값을 찾지 못함 러닝레이트가 너무커서 
#lr = 0.01
#loss: nan 결과물:  [[nan]]
#lr = 0.001
#loss: 4.9276831504130314e-08 결과물:  [[10.999633]]
#lr = 0.0001
#loss: 0.0009612962603569031 결과물:  [[10.958887]] #러닝레이트 너무줄이니까 또  loss 커지려함.

################Nadam########################### (lr=0.001)
#lr = 0.1
#loss: 103707.3203125 결과물:  [[188.58049]]
#lr = 0.01
#loss: 1738.517333984375 결과물:  [[13.474728]]
#lr = 0.001
#loss: 1.716671292964489e-12 결과물:  [[10.999995]]


from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.data
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]

#And gate

#2. model

#model = LinearSVC()
#model = SVC()

model = Sequential()
model.add(Dense(10,input_shape=(2,),activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1,activation='relu'))
#model.score: 1.0-> sigmoid-=0.5 relu -> 1
#3. compile,trainning
model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['acc'])
model.fit(x_data,y_data,batch_size=1,epochs=100)


#4. evaluate, predict
y_pred = model.predict(x_data)
print(x_data, "predict: ",y_pred)

result = model.evaluate(x_data,y_data)
print("model.score:", result[1])  #acuuracy #model.score: 0.75

# acc = accuracy_score(y_data, y_pred)
# print(acc) #0.75

#model.score: 1.0
#1.0
#acc:  0.75-> 1 (model : LinearSVC->SVC)

#model.score: 0.5 (no hidden layer)
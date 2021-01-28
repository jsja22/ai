from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import accuracy_score
#1.data
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,1]

#And gate

#2. model

model = LinearSVC()

#3. trainning

model.fit(x_data,y_data)


#4. evaluate, predict
y_pred = model.predict(x_data)
print(x_data, "predict: ",y_pred)

result = model.score(x_data,y_data)
print("model.score:", result)  #acuuracy #model.score: 1.0

acc = accuracy_score(y_data, y_pred)
print(acc) #1.0
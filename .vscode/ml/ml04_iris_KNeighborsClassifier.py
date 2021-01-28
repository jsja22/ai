import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris   #다중 분류모델
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
dataset = load_iris()

x = dataset.data
y= dataset.target
# from sklearn.preprocessing import MinMaxScaler #x값을 minmax전처리 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=66)
# #x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, train_size=0.8, shuffle=True)

# scalar = MinMaxScaler()
# scalar.fit(x_train)
# x_train = scalar.transform(x_train)
# x_test = scalar.transform(x_test)
# #x_val = scalar.transform(x_val)
'''
#원핫 인코딩  (다중분류에 인해서 y를 원핫인코딩한것이다.=>y를 벡터화 하는것) 값이 있는부분에만 1을 넣고 나머지엔 0 
#다중분류에서 y는 반드시 원핫인코딩 해줘야함
from tensorflow.keras.utils import to_categorical

y=to_categorical(y)
y_train= to_categorical(y_train)
y_test = to_categorical(y_test)

print(y)
print(x.shape) #(150, 4)
print(y.shape) #(150, 3)
'''
'''
print(dataset.DESCR)
print(dataset.feature_names) 

print(x.shape)  #(150,4)
print(y.shape)  #(150, )
print(x[:5])
print(y)
'''
model = KNeighborsClassifier(n_neighbors =3)


#model = LinearSVC()
# model = Sequential()
# model.add(Dense(10, activation='relu', input_shape=(4,)))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(3, activation='softmax'))  #softmax : output node 다합치면 1이되는함수, 3인이유는 y.shape가 (150,3)이기에 3 softmax는 다중분류에서만 사용
#                                           #가장 큰 값이 결정된다. 
# model.summary()
 
#컴파일,훈련
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])  #다중분류에서는 loss='categorical_crossentropy'사용 
from tensorflow.keras.callbacks import EarlyStopping
#early_stopping = EarlyStopping(monitor='loss',patience=30,mode='auto')
# model.fit(x_train,y_train, epochs=110, batch_size=8,validation_data=(x_test,y_test),verbose=0,callbacks=[early_stopping])
model.fit(x_train,y_train)
# loss = model.evaluate(x_test,y_test,batch_size=8)
result = model.score(x_test,y_test)
print(result) #(loss,acuracy)
 
print("test predict: {}".format(model.predict(x_test))) #test predict: [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 2 2 2 0 1 1 2 1 2 0 0 2 2]
print("test accuracy: {:.2f}".format(result))  #test accuracy: 0.97


# fig, axes = plt.subplot(1,3,figsize =(10,3))

# for n_neighbors, ax in zip([1,3,9], axes):
#     model = KNeighborsClassifier(n_neighbors = n_neighbors).fit(x,y)
#     mglearn.plots.plot_2d_separator(model,x,fill=True, eps=0.5, ax=ax, alpha=.4)
#     mglearn.discrete_scatter(x[:,0],x[:,1],y,ax=ax)
#     ax.set_title("{} neighbors".format(n_neighbors))
#     ax.set_xlabel("feature 0")
#     ax.set_ylabel("feature 1")
# axes[0].legend(loc=3)
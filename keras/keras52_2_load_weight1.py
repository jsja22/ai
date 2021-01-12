
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import mnist



(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(-1,28,28,1)/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


model = Sequential()
model.add(Conv2D(filters = 256,kernel_size = (2,2),padding='same',strides = 1,input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(256,kernel_size=(2,2)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='softmax'))

#model.save('C:/data/h5/k52_1_model1.h5')
#model.save_weights('C:/data/h5/k52_1_weight.h5')

#model1 = load_model('C:/data/h5/k52_1_model2.h5')
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#modelpath = 'C:/data/modelcheckpoint/k52_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
#cp = ModelCheckpoint(filepath = modelpath, monitor='val_loss',save_best_only = True, mode = 'auto')
#es = EarlyStopping(monitor='val_loss',patience=10)
# hist = model.fit(x_train,y_train,validation_split = 0.2,epochs=50,verbose=1,batch_size=16,callbacks=[es])#, CP])


#model.save('C:/data/h5/k52_1_model2.h5')

#4-1 evaluate, predict
#result1 = model1.evaluate(x_test,y_test,batch_size=16)

#print('model1_loss : ',result1[0])
#print('model1_accuracy : ',result1[1])

#4-2 evaluate, predict
model.load_weights('C:/data/h5/k52_1_weight.h5')

result1 = model.evaluate(x_test,y_test,batch_size=16)

print('model_loss : ',result1[0])
print('model_accuracy : ',result1[1])

########################
#52_1 save 
#모델을 save한것과 weight를 save한거 두개의 가중치가 같다.

#52_1 
#model save, weight save, check save 까지

#52_2 weight를 불러옴  (weight는 훈련의 결과)
########################weight가 있으면 훈련을 할필요없으니 compile,fit 모두 필요없어짐

#훈련이 필요없으니 loss,acureacy 바로 계산
#model_loss :  0.07627488672733307
#model_accuracy :  0.9843000173568726

model2 = load_model('C:/data/h5/k52_1_model2.h5')   # load model 한것   #model1 -> weight model 
result2= model2.evaluate(x_test,y_test,batch_size=8)
print("loadmodel_loss: ", result2[0])
print("loadmodel_accuracy: ", result2[1])

#save model 과 load model의 가중치 값이 같다
#loadmodel_loss:  0.0762747973203659
#loadmodel_accuracy:  0.9843000173568726

#만약에 weight값만 필요하면 save weight
#모델만 필요하고 다시 훈련하고 싶다 그러면 savemodel을 fit하기전에 함
#모델부터 weight까지 다 필요하다 save모델을 훈련시킨다음에 save 모델을 하면된다.


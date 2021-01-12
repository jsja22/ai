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

model = load_model('C:/data/h5/k51_1_model1.h5')

model.summary()

#model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#modelpath = './modelCheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
#cp = ModelCheckpoint(filepath = modelpath, monitor='val_loss',save_best_only = True, mode = 'auto')
#es = EarlyStopping(monitor='val_loss',patience=10)
#hist = model.fit(x_train,y_train,validation_split = 0.2,epochs=50,verbose=1,batch_size=16,callbacks=[es,cp])    # if this model is best model, we dont need train anymore

model = load_model('C:/data/h5/k51_1_model2.h5')     #after weight value save (compile,fit) 

result = model.evaluate(x_test,y_test,batch_size=16)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis=-1)
print('loss : ',result[0])
print('accuracy : ',result[1])

#loss :  0.0838533341884613
#accuracy :  0.9830999970436096
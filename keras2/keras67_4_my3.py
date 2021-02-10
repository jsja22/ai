
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv2D,Dense, Dropout, MaxPooling2D,Flatten, BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import os
import numpy as np
from PIL import Image

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=0.7,
    fill_mode='nearest',
    validation_split=0.2
)
test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    'C:/data/image/sex',
    target_size=(150,150),
    batch_size=14,
    class_mode='binary',
    subset='training'
)
xy_test = train_datagen.flow_from_directory(
    'C:/data/image/sex',  #마지막 경로가 반드시 폴더여야함
    target_size=(150,150), #size
    batch_size=14,
    class_mode='binary',
    subset='validation'
)
x_pred = train_datagen.flow_from_directory(
    'C:/data/image/my/',
    target_size = (150,150),
    batch_size= 14,
    class_mode='binary', 
)
print(xy_train)  
print(xy_test)
print(xy_train[0][0].shape) 
print(xy_train[0][1].shape) 

# Found 1389 images belonging to 1 classes.
# Found 347 images belonging to 1 classes.
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x00000224903B8550>
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000022490439160>
# (14, 64, 64, 3)
# (14,)

model = Sequential()
model.add(Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(150,150,3)))
model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])
#print(xy_train.samples) #1389
cp = ModelCheckpoint('C:/data/modelcheckpoint/my2.hdf5') 
es = EarlyStopping(monitor = 'val_loss', patience = 15)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 6, factor = 0.5, verbose = 1)
history = model.fit_generator(xy_train, steps_per_epoch=1389/14 ,epochs=100, validation_data=xy_test, validation_steps=1389/14,callbacks=[es,lr,cp])
'''
model2 = load_model('C:/data/modelcheckpoint/my2.hdf5', compile=False)
# model.save_weights('C:/data/h5/k67_4_weight.h5')
# model.load_weights('C:/data/h5/k67_4_weight.h5')

result = model.predict_generator(x_pred,verbose=True)
print(result)
print("남자일 확률은",result*100,"%입니다.")

for i in range(2):
    filepath = 'C:/data/image/my2/%d.jpg'%i
    img = Image.open(filepath)
    img = img.convert("RGB")
    img = img.resize((150,150))
    data = np.asarray(img)
    X = np.array(data)
    X = X.astype("float")/256
    print(X.shape) #(150, 150, 3)
    X = X.reshape(-1,150,150,3)
    print(X.shape)
    categories = ["man","girl"]
    x_pred = test_datagen.flow(X)
    y_pred = model.predict(x_pred)
    y_pred = y_pred[0][0]
    print(y_pred)
    # result = [np.argmax(value) for value in y_pred] # 예측 값중 가장 높은 클래스 반환
    # print(result)
    print('photo : ',categories[i])
    if y_pred >0.5:
        print("남자일 확률 ",np.round(y_pred*100,2), '%')
    elif y_pred <0.5:
        print("여자일 확률 ",np.round((1-y_pred)*100,2), '%')
    elif y_pred =0.5:
        print("중성입니다.")
    # print('New image prediction : ',categories[result[0]])
    # #print("accuracy : {}".format(max(pred[0][0],pred[0][1])))
    # print("accuracy : {}".format(max(pred[0])))
    plt.subplot(2,1,i+1)
    plt.imshow(img)
plt.show()
'''
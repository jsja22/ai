import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import EfficientNetB3
import gc
import cv2
from keras import backend as bek
import warnings
def effnet():
    effnet = EfficientNetB3(include_top=True,  weights=None,input_shape=(300,300,3), classes=10, classifier_activation="softmax",)

    model = Sequential()
    model.add(effnet)
    print("################")
    return model
warnings.filterwarnings('ignore')
train = pd.read_csv('C:/data/computer vision/mnist_data/train.csv',index_col=None, header=0)
test = pd.read_csv('C:/data/computer vision/mnist_data/test.csv',index_col=None, header=0)
sub = pd.read_csv('C:/data/computer vision/mnist_data/submission.csv',index_col=None, header=0)

print(train.shape, test.shape) #(2048, 787) (20480, 786)
print(sub.shape) #[20480 rows x 2 columns]
print(train,test,sub) 

train['digit'].value_counts() # value_counts() ->어떤 컬럼/Series의 unique value들을 count해주는 함수
# drop columns
train2 = train.drop(['id','digit','letter'],1)
test2 = test.drop(['id','letter'],1)  

# convert pandas dataframe to numpy array
train2 = train2.values
test2 = test2.values

plt.imshow(train2[2047].reshape(28,28))

# reshape
train2 = train2.reshape(-1,28,28,1)
test2 = test2.reshape(-1,28,28,1)

# data normalization
train2 = train2/255.0
test2 = test2/255.0

train_224=np.zeros([2048,300,300,3],dtype=np.float32)



'''
# ImageDatagenerator & data augmentation
idg = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1)) #augmentation 인자를 지정
idg2 = ImageDataGenerator()

# show augmented image data
sample_data = train2[2047].copy()
print(sample_data.shape)#(28,28,1)
sample = expand_dims(sample_data,axis=0)  #expand_dims차원늘려주기 axis = 0 이면 (1,28,28,1) axis = 1이면 (28, 1, 28, 1)이런식
sample_datagen = ImageDataGenerator(height_shift_range=(-1,1), width_shift_range=(-1,1))
sample_generator = sample_datagen.flow(sample, batch_size=1)

plt.figure(figsize=(16,10))

for i in range(9) : 
    plt.subplot(3,3,i+1)
    sample_batch = sample_generator.next()
    sample_image=sample_batch[0]
    plt.imshow(sample_image.reshape(28,28))
plt.show()

# cross validation
skf = StratifiedKFold(n_splits=40, random_state=42, shuffle=True)

reLR = ReduceLROnPlateau(patience=30,verbose=1,factor=0.5) #learning rate scheduler
es = EarlyStopping(patience=60, verbose=1,mode='auto')

val_loss_min = []
result = 0
nth = 0

for train_index, valid_index in skf.split(train2,train['digit']) :
    
    mc = ModelCheckpoint('C:/data/computer vision/h5/Dacon_computer_vision_0203_1.h5',save_best_only=True, verbose=1)
    
    x_train = train2[train_index]
    x_valid = train2[valid_index]    
    y_train = train['digit'][train_index]
    y_valid = train['digit'][valid_index]
    
    train_generator = idg.flow(x_train,y_train,batch_size=8)
    valid_generator = idg2.flow(x_valid,y_valid)
    test_generator = idg2.flow(test2,shuffle=False)
    
    model = effnet()

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.002,epsilon=None),metrics=['acc'])
    
    learning_history = model.fit_generator(train_generator,epochs=1000, validation_data=valid_generator, callbacks=[es,mc,reLR])
    
    # predict
    model.load_weights('C:/data/computer vision/h5/Dacon_computer_vision_0203_1.h5')
    result += model.predict_generator(test_generator,verbose=True)/40
    
    # save val_loss
    hist = pd.DataFrame(learning_history.history)
    val_loss_min.append(hist['val_loss'].min())
    
    nth += 1
    print(nth, '번째 학습을 완료했습니다.')

sub['digit'] = result.argmax(1)
sub.to_csv('C:/data/computer vision/csv/Dacon_computer_vision_0203_1.csv',index=False)
'''
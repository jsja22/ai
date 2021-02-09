import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.7,
    fill_mode='nearest'
)

train_datagen = ImageDataGenerator(rescale=1./255) #test는 rescale만
0
#flow or flow_from_directory  -> 데이터로 변환

#이미 수치화 되어있는걸 가져오려면 그냥 flow
#train_generator
#train에 ad,normal 이있기때문에 y는 0과 1
xy_train = train_datagen.flow_from_directory(
    'C:/data/image/brain/data1/train',
    target_size=(150,150), #주고싶은 사이즈로 줄이거나 늘임
    batch_size=30,
    class_mode='binary',
    save_to_dir='C:/data/image/brain_generator/train'
)
#(80,150,150,1)
#test_generator
xy_test = train_datagen.flow_from_directory(
    'C:/data/image/brain/data1/test',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary'
)

print(xy_train[0][0])
print(xy_train[0][1])
print(xy_train[0][1].shape)
print(xy_train[1][1])  #batchsize x print문 갯수 만큼 생성됨 #batch_size 160이 최대
#flow로 해보는것이 과제!

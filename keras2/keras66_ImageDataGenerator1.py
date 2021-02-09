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

#flow or flow_from_directory  -> 데이터로 변환

#이미 수치화 되어있는걸 가져오려면 그냥 flow
#train_generator
#train에 ad,normal 이있기때문에 y는 0과 1
xy_train = train_datagen.flow_from_directory(
    'C:/data/image/brain/data1/train',
    target_size=(150,150), #주고싶은 사이즈로 줄이거나 늘임
    batch_size=5,
    class_mode='binary'
)
#(80,150,150,1)
#test_generator
xy_test = train_datagen.flow_from_directory(
    'C:/data/image/brain/data1/test',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary'
)

print(xy_train)  
print(xy_test)
#Found 160 images belonging to 2 classes. 2개의 이미지 80개씩
#Found 120 images belonging to 2 classes.
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000027F679385B0>
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000027F679493D0>

print(xy_train[0]) #0번째의 0번째가 x 0번째의 1번째가 y
print(xy_train[0][0])
print(xy_train[0][0].shape) #(5, 150, 150, 3)  -> x  5->batch_size
print(xy_train[0][1]) #[0. 0. 1. 0. 0.] ->y 5개
print(xy_train[0][1].shape) #(5,)
print(xy_train[31][1])  #160장의데이터를 5로 나눴기때문에 0부터 31번째 까지 있다!

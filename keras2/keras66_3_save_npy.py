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
    batch_size=160,
    class_mode='binary'
)
#(80,150,150,1)
#test_generator
xy_test = train_datagen.flow_from_directory(
    'C:/data/image/brain/data1/test',
    target_size=(150,150),
    batch_size=120,
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
print(xy_train[0][0].shape) #(160, 150, 150, 3)
print(xy_train[0][1])
print(xy_train[0][1].shape) #(160,)

np.save('C:/data/image/brain/data1/npy/keras66_train_x.npy',arr=xy_train[0][0])
np.save('C:/data/image/brain/data1/npy/keras66_train_y.npy',arr=xy_train[0][1])
np.save('C:/data/image/brain/data1/npy/keras66_test_x.npy',arr=xy_test[0][0])
np.save('C:/data/image/brain/data1/npy/keras66_test_y.npy',arr=xy_test[0][1])

x_train = np.load('C:/data/image/brain/data1/npy/keras66_train_x.npy')
y_train = np.load('C:/data/image/brain/data1/npy/keras66_train_y.npy')
x_test = np.load('C:/data/image/brain/data1/npy/keras66_test_x.npy')
y_test = np.load('C:/data/image/brain/data1/npy/keras66_test_y.npy')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# (160, 150, 150, 3) (160,)
# (120, 150, 150, 3) (120,)
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#flow로 구성하기 
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

test_datagen = ImageDataGenerator(rescale=1./255) #test는 rescale만
#npy 불러와야함
x_train = np.load('C:/data/image/brain/data1/npy/keras66_train_x.npy')
y_train = np.load('C:/data/image/brain/data1/npy/keras66_train_y.npy')
x_test = np.load('C:/data/image/brain/data1/npy/keras66_test_x.npy')
y_test = np.load('C:/data/image/brain/data1/npy/keras66_test_y.npy')

batch_size=5
seed=66
train_generator = train_datagen.flow(x_train,y_train,batch_size,seed)
test_generator = test_datagen.flow(x_test,y_test,batch_size,seed)

print(train_generator[0][0].shape)
print(train_generator[0][1].shape)
print(test_generator[0][0].shape)
print(test_generator[0][1].shape)

# (5, 150, 150, 3)
# (5,)
# (5, 150, 150, 3)
# (5,)

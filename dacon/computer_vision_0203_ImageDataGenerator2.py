from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode = 'nearest'
)

img = load_img('C:/data/computer vision/test_dirty_mnist_2nd/54999.png')
x = img_to_array(img)
print(x.shape)
x= x.reshape((1,)+x.shape) 
print(x.shape) #(1, 256, 256, 3) #4차원 변환

i = 0
for batch in datagen.flow(x,batch_size=1, save_to_dir='C:/data/computer vision/',save_prefix='54999',save_format='jpeg'):
    i+=1
    if i>20:
        break
#Generator를 이용해서 이미지 augmentation을 수행하는 목적이 바로 이런 과적합을 막는 데에 있음.
#오히려 모델의 복잡도를 줄이면 가장 중요한 몇 가지 특징에 집중해서 분류 대상을 정확하게 파악하고, 새로운 이미지도 정확하게 분류해낼 수 있게 됨.

from sklearn.keras.model import Sequential
from sklearn.kear.layers import Conv2D, MaxPooling2D, BatchNormalization,Dropout
from sklearn.kear.layers import Flatten

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax')

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#flow_from_directory() 함수를 사용하면 이미지가 저장된 폴더를 기준으로 라벨 정보와 함께 이미지를 불러올 수 있다!

batch_size = 16

# 학습 이미지에 적용한 augmentation 인자를 지정해줍니다.
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
# 검증 및 테스트 이미지는 augmentation을 적용하지 않습니다. 모델 성능을 평가할 때에는 이미지 원본을 사용합니다.
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode=`nearest`)
# 앞서 언급했듯이 ImageDataGenerator는 이미지 augmentation을 도와주는 친구입니다. 이미지를 마음대로 바꾸면 곤란하니까 저희가 몇 가지 지시 사항을 줄 것입니다. 각각의 인자가 어떤 사항을 지시하는지 설명드리겠습니다.

# rotation_range: 이미지 회전 범위 (degrees)
# width_shift, height_shift: 그림을 수평 또는 수직으로 랜덤하게 평행 이동시키는 범위 (원본 가로, 세로 길이에 대한 비율 값)
# rescale: 원본 영상은 0-255의 RGB 계수로 구성되는데, 이 같은 입력값은 모델을 효과적으로 학습시키기에 너무 높습니다 (통상적인 learning rate를 사용할 경우). 그래서 이를 1/255로 스케일링하여 0-1 범위로 변환시켜줍니다. 이는 다른 전처리 과정에 앞서 가장 먼저 적용됩니다.
# shear_range: 임의 전단 변환 (shearing transformation) 범위
# zoom_range: 임의 확대/축소 범위
# horizontal_flip: True로 설정할 경우, 50% 확률로 이미지를 수평으로 뒤집습니다. 원본 이미지에 수평 비대칭성이 없을 때 효과적입니다. 즉, 뒤집어도 자연스러울 때 사용하면 좋습니다.
# fill_mode 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
# 케라스 공식 문서를 보시면 이외에도 다양한 인자가 있습니다.

# ImageDataGenerator 디버깅
# Generator를 이용해서 학습하기 전, 먼저 변형된 이미지에 이상한 점이 없는지 확인해야 합니다. 케라스는 이를 돕기 위해 flow라는 함수를 제공합니다. 여기서 rescale 인자는 빼고 진행합니다—255배 어두운 이미지는 아무래도 눈으로 확인하기 힘들 테니까 말이죠.

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode=`nearest`)

img = load_img(`data/train/cats/cat.0.jpg`)  # PIL 이미지
x = img_to_array(img)  # (3, 150, 150) 크기의 NumPy 배열
x = x.reshape((1,) + x.shape)  # (1, 3, 150, 150) 크기의 NumPy 배열

# 아래 .flow() 함수는 임의 변환된 이미지를 배치 단위로 생성해서
# 지정된 `preview/` 폴더에 저장합니다.
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=`preview`, save_prefix=`cat`, save_format=`jpeg`):
    i += 1
    if i > 20:
        break  # 이미지 20장을 생성하고 마칩니다

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150)))
model.add(Activation(`relu`))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation(`relu`))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation(`relu`))
model.add(MaxPooling2D(pool_size=(2, 2)))
모델의 상단에는 두 개의 fully-connected 레이어를 배치할 것입니다. 마지막으로 sigmoid 활성화 레이어를 배치합니다. 만약 분류하고자 하는 클래스가 3가지 이상이면 softmax 활성화 레이어를 사용하시면 됩니다. 손실 함수는 binary_crossentropy를 사용합니다. 이 또한 클래스가 3가지 이상이면 categorical_crossentropy를 사용하시면 됩니다.

model.add(Flatten())  # 이전 CNN 레이어에서 나온 3차원 배열은 1차원으로 뽑아줍니다
model.add(Dense(64))
model.add(Activation(`relu`))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation(`sigmoid`))

model.compile(loss=`binary_crossentropy`,
              optimizer=`rmsprop`,
              metrics=[`accuracy`])

batch_size = 16

# 학습 이미지에 적용한 augmentation 인자를 지정해줍니다.
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# 검증 및 테스트 이미지는 augmentation을 적용하지 않습니다. 모델 성능을 평가할 때에는 이미지 원본을 사용합니다.
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 이미지를 배치 단위로 불러와 줄 generator입니다.
train_generator = train_datagen.flow_from_directory(
        `data/train`,  # this is the target directory
        target_size=(150, 150),  # 모든 이미지의 크기가 150x150로 조정됩니다.
        batch_size=batch_size,
        class_mode=`binary`)  # binary_crossentropy 손실 함수를 사용하므로 binary 형태로 라벨을 불러와야 합니다.

validation_generator = validation_datagen.flow_from_directory(
        `data/validation`,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode=`binary`)

test_generator = test_datagen.flow_from_directory(
        `data/validation`,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode=`binary`)

        model.fit_generator(
        train_generator,
        steps_per_epoch=1000 // batch_size,
        validation_data=validation_generator,
        epochs=50)
model.save_weights(`first_try.h5`)  # 많은 시간을 들여 학습한 모델인 만큼, 학습 후에는 꼭 모델을 저장해줍시다.


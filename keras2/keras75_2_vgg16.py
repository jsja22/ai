from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

vgg16= VGG16(weights='imagenet', include_top= False, input_shape=(32,32,3))


vgg16.trainable=False #훈련시키지 않겠다는 뜻
# Total params: 14,714,688
# Trainable params: 0
# Non-trainable params: 14,714,688

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))
model.summary()
# Total params: 14,719,879
# Trainable params: 5,191   -> 이것만 훈련시키겠다
# Non-trainable params: 14,714,688


print("original weight number:",len(model.weights)) #26 -> 32
print("동결(freezon) 전 훈련되는 가중치의 수",len(model.trainable_weights)) #26  #fasle->0->6

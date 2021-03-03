from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

model= VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))


model.trainable=False #훈련시키지 않겠다는 뜻
# Total params: 14,714,688
# Trainable params: 0
# Non-trainable params: 14,714,688


model.summary()

print("original weight number:",len(model.weights)) 
print("동결(freezon) 전 훈련되는 가중치의 수",len(model.trainable_weights))

#include_top(True)
# Total params: 138,357,544
# Trainable params: 0
# Non-trainable params: 138,357,544
# _________________________________________________________________
# original weight number: 32
# 동결(freezon) 전 훈련되는 가중치의 수 0

#include_top(False)

# Total params: 14,714,688
# Trainable params: 0
# Non-trainable params: 14,714,688
# _________________________________________________________________
# original weight number: 26
# 동결(freezon) 전 훈련되는 가중치의 수 0

#INCLUDE_TOP = True 하면 FullyConnect까지 그대로 들어가야함
#False 하면 inpuut에 커스터마이징 또한 flatten 이후 layer를 커스터 마이징 할 수 있
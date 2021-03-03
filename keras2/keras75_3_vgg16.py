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

################################
import pandas as pd
pd.set_option('max_colwidth',-1)
layers = [(layer,layer.name,layer.trainable) for layer in model.layers]
aaa = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])


print(aaa)
#                                                                             Layer Type Layer Name  Layer Trainable
# 0  <tensorflow.python.keras.engine.functional.Functional object at 0x0000020B0B6AC550>  vgg16      False
# 1  <tensorflow.python.keras.layers.core.Flatten object at 0x0000020B0B740C10>           flatten    True
# 2  <tensorflow.python.keras.layers.core.Dense object at 0x0000020B0B75B520>             dense      True
# 3  <tensorflow.python.keras.layers.core.Dense object at 0x0000020B0B773A90>             dense_1    True
# 4  <tensorflow.python.keras.layers.core.Dense object at 0x0000020B0B784A30>             dense_2    True
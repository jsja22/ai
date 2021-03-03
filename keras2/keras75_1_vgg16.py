from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

model = VGG16(weights='imagenet', include_top= False, input_shape=(32,32,3))
print(model.weights)

model.trainable=False #훈련시키지 않겠다는 뜻

model.summary()

print(len(model.weights)) #26
print(len(model.trainable_weights)) #26  #fasle->0


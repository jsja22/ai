from tensorflow.keras.applications import VGG16,VGG19,Xception
from tensorflow.keras.applications import ResNet101,ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet,MobileNetV2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0,EfficientNetB1

# model = VGG19()

# model.trainable=True

# model.summary()
# print(len(model.weights)) #38
# print(len(model.trainable_weights)) #38


# model = Xception()

# model.trainable=True

# model.summary()
# print(len(model.weights)) #236
# print(len(model.trainable_weights)) #156


model = EfficientNetB0()

model.trainable=True

model.summary()
print(len(model.weights)) #236
print(len(model.trainable_weights)) #156
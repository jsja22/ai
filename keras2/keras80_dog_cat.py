# data/image/vgg  4 
#개,고양이, 라이언, 슈트
#dog1.jpg,cat1.jg,lion1.jpg,suit1.jpg

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

img_dog = load_img('C:/data/vgg/dog1.jpg ', target_size =(224,224))
img_cat = load_img('C:/data/vgg/cat1.jpg ', target_size =(224,224))
img_lion = load_img('C:/data/vgg/lion1.jpg ', target_size =(224,224))
img_suit = load_img('C:/data/vgg/suit1.jpg ', target_size =(224,224))
print(img_dog)
arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_lion = img_to_array(img_lion)
arr_suit = img_to_array(img_suit)
print(arr_dog)
print(type(arr_dog))
print(arr_dog.shape)  #(224,224,3)

from tensorflow.keras.applications.vgg16 import preprocess_input
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_lion = preprocess_input(arr_lion)
arr_suit = preprocess_input(arr_suit)
print(arr_dog)
print(arr_dog.shape)


arr_input = np.stack([arr_dog,arr_cat,arr_lion,arr_suit]) #(4,224,224,3)
model = VGG16()
results = model.predict(arr_input)

print(results)
print('results.shape:', results.shape) #results.shape: (4, 1000)

from tensorflow.keras.applications.vgg16 import decode_predictions

decoded = decode_predictions(results)
print("decoded[0] : ", decoded[0])
print("=========================")
print("decoded[1] : ", decoded[1])
print("=========================")
print("decoded[2] : ", decoded[2])
print("=========================")
print("decoded[3] : ", decoded[3])


# # 위노그라드 알고리즘 설정 (GPU 사용시 conv 연산이 빨라짐)
# os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

# rootPath = './datasets/cat-and-dog'

# # 이미지의 사이즈는 논문에서는 (224, 224, 3)을 사용하여, 빠른 학습을 위해 사이즈를 조정
# IMG_SIZE = (150, 150, 3)  

# imageGenerator = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     brightness_range=[.2, .2],
#     horizontal_flip=True
# )

# trainGen = imageGenerator.flow_from_directory(
#     os.path.join(rootPath, 'training_set'),
#     target_size=IMG_SIZE[:2],
#     batch_size=20
# )

# testGen = ImageDataGenerator(
#     rescale=1./255,
# ).flow_from_directory(
#     os.path.join(rootPath, 'test_set'),
#     target_size=IMG_SIZE[:2],
#     batch_size=20,
# )



# from tensorflow.keras import layers
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.applications import InceptionV3

# extractor = Sequential()
# extractor.add(InceptionV3(include_top=False, weights='imagenet', input_shape=IMG_SIZE))
# extractor.add(layers.GlobalAveragePooling2D())

# extractor_output_shape = extractor.get_output_shape_at(0)[1:]

# model = Sequential()
# model.add(layers.InputLayer(input_shape=extractor_output_shape))
# model.add(layers.Dense(2, activation='sigmoid'))

# model.summary()

# from tensorflow.keras.optimizers import Adam
# optimizer = Adam(lr=0.001)
# model.compile(
#     optimizer=optimizer,
#     loss='binary_crossentropy', 
#     metrics=['acc'],
# )

# extractor.compile(
#     optimizer=optimizer,
#     loss='binary_crossentropy', 
#     metrics=['acc'],
# )
# '''
# import matplotlib.pyplot as plt

# def show_graph(history_dict):
#     accuracy = history_dict['acc']
#     val_accuracy = history_dict['val_acc']
#     loss = history_dict['loss']
#     val_loss = history_dict['val_loss']

#     epochs = range(1, len(loss) + 1)
    
#     plt.figure(figsize=(16, 1))
    
#     plt.subplot(121)
#     plt.subplots_adjust(top=2)
#     plt.plot(epochs, accuracy, 'ro', label='Training accuracy')
#     plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
#     plt.title('Trainging and validation accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')

#     plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
#               fancybox=True, shadow=True, ncol=5)

#     plt.subplot(122)
#     plt.plot(epochs, loss, 'bo', label='Training loss')
#     plt.plot(epochs, val_loss, 'b', label='Validation loss')
#     plt.title('Training and validation loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
#           fancybox=True, shadow=True, ncol=5)

#     plt.show()
    
# def smooth_curve(points, factor=.8):
#     smoothed_points = []
#     for point in points:
#         if smoothed_points:
#             previous = smoothed_points[-1]
#             smoothed_points.append(previous * factor + point * (1 - factor))
#         else:
#             smoothed_points.append(point)
#     return smoothed_points

# smooth_data = {}
# for key, val in history.history.items():
#     smooth_data[key] = smooth_curve(val[:])
# show_graph(smooth_data)

# '''
# model.evaluate(testX, testY)

import numpy as np
import PIL
from numpy import asarray
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, KFold
from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
import string
import scipy.signal as signal
from keras.applications.resnet import ResNet101,preprocess_input
from tqdm import tqdm

img1=[]
for i in tqdm(range(0,72000)):
    filepath='C:/data/LPD_competition/pred/test/%d.jpg'%i
    image2=Image.open(filepath)
    image2 = image2.convert('RGB')
    image2 = image2.resize((224,224))
    image_data2=asarray(image2)
    # image_data2 = signal.medfilt2d(np.array(image_data2), kernel_size=3)
    img1.append(image_data2)    

# np.save('../data/csv/Dacon3/train4.npy', arr=img)
np.save('C:/data/LPD_competition/npy/test_224.npy', arr=img1)
# alphabets = string.ascii_lowercase
# alphabets = list(alphabets)


# x = np.load('../data/csv/Dacon3/train4.npy')
x_pred = np.load('C:/data/LPD_competition/npy/test_224.npy',allow_pickle=True)

print(x_pred.shape) #(72000, 255, 255, 3)
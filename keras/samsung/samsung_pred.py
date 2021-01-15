import numpy as np
import pandas as pd

x_train = np.load('C:/data/npy/samsung_x_test.npy')
x_test = np.load('C:/data/npy/samsung_x_train.npy')
y_train = np.load('C:/data/npy/samsung_y_test.npy')
y_test = np.load('C:/data/npy/samsung_y_train.npy')
x_data = np.load('C:/data/npy/samsung_x_data.npy')
from tensorflow.keras.models import load_model

model = load_model('C:/data/h5/samsung_stock.h5')

loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_pred1 = x_data[-1].reshape(1,x_train.shape[1],x_train.shape[2])
value = model.predict(y_pred1)
print('종가= ', value)

종가=  [[88400.414]]
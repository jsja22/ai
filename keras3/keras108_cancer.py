import numpy as np
import tensorflow as tf
import autokeras as ak

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)

print(x_train.shape, x_test. shape) 
print(y_train.shape, y_test.shape) 

scale = MinMaxScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)


#ImageRegressor는 shape가 맞지않음
model = ak.StructuredDataClassifier(
    overwrite=True,
    max_trials=2,
    #loss='mse',
    #metrics=['acc']
)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

es= EarlyStopping(monitor='val_loss',mode='min', patience=6)
lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=2)
ck = ModelCheckpoint('C:/data/modelcheckpoint/', save_best_only=True, save_weights_only=True, monitor='val_loss', verbose=1)


model.fit(x_train, y_train, epochs=10, validation_split=0.2, callbacks=[es, lr, ck])

results = model.evaluate(x_test, y_test)
print(results)  

model2 = model.export_model()
try:
    model2.save('C:/data/save/cancer', save_format='tf')
except:
    model2.save('C:/data/save/cancer.h5')

# 4/4 [==============================] - 0s 4ms/step - loss: 0.0745 - accuracy: 0.9737
# [0.07452034205198288, 0.9736841917037964]
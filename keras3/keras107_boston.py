import numpy as np
import tensorflow as tf
import autokeras as ak

from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)

print(x_train.shape, x_test. shape) # (404, 13) (102, 13)
print(y_train.shape, y_test.shape)  # (404,) (102,)  

scale = StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)


#ImageRegressor응 shape가 맞지않음
model = ak.StructuredDataRegressor(
    overwrite=True,
    max_trials=2,
    metrics=['mse']
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
    model2.save('C:/data/save/boston', save_format='tf')
except:
    model2.save('C:/data/save/boston.h5')

from tensorflow.keras.models import load_model

model3 = load_model('C:/data/save/boston', custom_objects=ak.CUSTOM_OBJECTS)
result_boston = model3.evaluate(x_test, y_test)

y_pred = model3.predict(x_test)
r2 = r2_score(y_test, y_pred)

print("load_result :", result_boston, r2)

#load_result : [93.06444549560547, 93.06444549560547] -0.11343843631438677
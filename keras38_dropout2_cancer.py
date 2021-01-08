import numpy as np


# 1. 데이터
from sklearn.datasets import load_breast_cancer

datasets = load_breast_cancer()

print(datasets.feature_names)
print(datasets.DESCR)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)  #(569, 30)  (569,)
print(x[:5])
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(64, activation='sigmoid', input_shape=(30,)))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid')) # 이진분류일때 마지막 activation은 반드시 sigmoid

model.summary()
# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # 이진분류일때 loss는 binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
hist = model.fit(x_train, y_train, epochs=1000, validation_data=(x_val, y_val), callbacks=[early_stopping])


# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print(loss, acc)

# 실습 1. acc 0.985이상 올릴것
# 실습 2. predict 출력해 볼것


y_pred = model.predict(x[-5:-1])
print(y_pred)
print(y[-5:-1])
y_recovery = np.where(y_pred<0.5, 0, 1)
print(y_recovery)

# 드랍아웃전
# 0.06372705101966858 0.9824561476707458
# 드랍아웃 후
# 0.05475200340151787 0.9824561476707458
#=> acc 는 변화 없는데 loss값은 오히려 좋아짐 
# loss가 첫번째로 평가됨 acc보다는
# train에서는 80퍼만쓰고  test에서는 100퍼를 쓴다.따라서 summary에서 파라미터 수가 변하지 않는다. =>다르게 생각하면 max값 잡아 놓고 사용한다고 생각하면 됨.
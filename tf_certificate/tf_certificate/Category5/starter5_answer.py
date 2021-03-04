import csv
import tensorflow as tf
import numpy as np
import urllib

from tensorflow.keras.layers import Dense, LSTM, Lambda, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import Huber



def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1) 
    print(series.shape) #(3000, 1) (235, 1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    # drop_remainder 인자를 사용하면, 마지막 배치 크기를 무시하고 지정한 배치 크기를 사용할 수 있습니다.
    #31개씩 한묶음으로 한칸씩 이동하면서 데이셋 만들어줌
    #window_size + 1, 30일의 데이터를 보고 31일째의 데이터를 에측하기 위함 
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    #flat_map을 통해서 각 batch 별로 flatten하게 shape을 펼쳐줌
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    #train/label 이 섞여서 31개의 데이터가 각 batch에 잡힘 이를 train 30개 label 1개로 분리해주면됨
    for train, label in ds.take(2):
      print('train: {}'.format(train))
      print('label: {}'.format(label))
    return ds.batch(batch_size).prefetch(1)


url = 'https://storage.googleapis.com/download.tensorflow.org/data/Sunspots.csv'
urllib.request.urlretrieve(url, 'sunspots.csv')

time_step = []
sunspots = []

with open('sunspots.csv') as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  next(reader)
  for row in reader:
    #print(row) #['3024', '2001-01-31', '142.6']
    sunspots.append(float(row[2]))
    time_step.append(int(row[0]))

series = np.array(sunspots)
print(series.shape)
min = np.min(series)
max = np.max(series)
series -= min
series /= max
time = np.array(time_step)

split_time = 3000

time_train = time[:split_time]
time_valid = time[split_time:]
x_train = series[:split_time]
x_valid = series[split_time:]


window_size = 30
batch_size = 32
shuffle_buffer_size = 1000


train_set = windowed_dataset(x_train, window_size=window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)
validation_set = windowed_dataset(x_valid, window_size=window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)

print("train_set :",train_set) #<PrefetchDataset shapes: ((None, None, 1), (None, None, 1)), types: (tf.float64, tf.float64)>

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(128, kernel_size=5,
                     padding="causal",  #same mae: 0.1354 #causal 
                     activation="relu",
                     input_shape=[None, 1]),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.LSTM(32, return_sequences=True),
  tf.keras.layers.Dense(32, activation="relu"),
  tf.keras.layers.Dense(16, activation="relu"),
  tf.keras.layers.Dense(1)
])

optimizer = SGD(lr=1e-5, momentum=0.9)
loss= Huber()
model.compile(loss=loss,optimizer=optimizer,metrics=["mae"])
cp_path = 'tmp_checkpoint.ckpt'
cp = ModelCheckpoint(cp_path,
                            save_weights_only=True,
                            save_best_only=True,
                            monitor='val_mae',
                            verbose=1)
model.fit(train_set, validation_data=(validation_set), epochs=100, callbacks=[cp])
model.load_weights(cp_path)

# loss: 0.0127 - mae: 0.1258 - val_loss: 0.0087 - val_mae: 0.1093
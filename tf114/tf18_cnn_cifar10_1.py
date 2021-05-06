import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 1. data
tf.compat.v1.set_random_seed(66)

tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly()) #False

print(tf.__version__) #2.3.1

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, x_test.shape)                              # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape)                              # (50000, 1) (10000, 1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)                            # (50000, 10) (10000, 10)

# minmaxscaler
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
print(x_train.shape, x_test.shape)                              # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape)                              # (50000, 10) (10000, 10)

learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)  #60000/100
# 1-3. pre_model_params
x = tf.compat.v1.placeholder('float',[None,32,32,3])
y = tf.compat.v1.placeholder('float',[None,10])
keep_prob = tf.compat.v1.placeholder(tf.float32)                          # dropout

# 2. model
w1 = tf.compat.v1.get_variable('w1', shape=[3, 3, 3, 32])                 # [kernel_size, kernel_size, channel, output]
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L1)                                                       # (?, 16, 16, 32)

w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 32, 64])                # [ksize, ksize, input, output]
L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.selu(L2)
L2 = tf.nn.max_pool(L2, [1,2,2,1], [1,2,2,1], padding='SAME')
print(L2)                                                       # (?, 8, 8, 64)

w3 = tf.compat.v1.get_variable('w3', shape=[3, 3, 64, 128])
L3 = tf.nn.conv2d(L2, w3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.selu(L3)
L3 = tf.nn.max_pool(L3, [1,2,2,1], [1,2,2,1], padding='SAME')
print(L3)                                                       # (?, 4, 4, 128)

w4 = tf.compat.v1.get_variable('w4', shape=[3, 3, 128, 64])
L4 = tf.nn.conv2d(L3, w4, strides=[1,1,1,1], padding='SAME')
L4 = tf.nn.selu(L4)
L4 = tf.nn.max_pool(L4, [1,2,2,1], [1,2,2,1], padding='SAME')
print(L4)                                                       # (?, 2, 2, 64)


# 2-1. flatten & dense layer & softmax output
L_flat = tf.reshape(L4, [-1, 2*2*64])
print(L_flat)

dw1 = tf.compat.v1.get_variable('w5', shape=[2*2*64, 64], initializer=tf.compat.v1.initializers.variance_scaling())
b1 = tf.Variable(tf.compat.v1.random_normal([64]), name = 'bias1')
dL1 = tf.nn.selu(tf.matmul(L_flat, dw1) + b1)

dw2 = tf.compat.v1.get_variable('w6', shape=[64, 32], initializer=tf.compat.v1.initializers.variance_scaling())
b2 = tf.Variable(tf.compat.v1.random_normal([32]), name = 'bias2')
dL2 = tf.nn.selu(tf.matmul(dL1, dw2) + b2)

dw3 = tf.compat.v1.get_variable('w7', shape=[32, 10], initializer=tf.compat.v1.initializers.variance_scaling())
b3 = tf.Variable(tf.compat.v1.random_normal([10]), name = 'bias3')
hypothesis = tf.nn.softmax(tf.matmul(dL2, dw3) + b3)


# 3. compile
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis = 1))
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00005).minimize(loss)

#훈련 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(training_epochs):
    avg_loss = 0

    for i in range(total_batch):  # 600번 돈다
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}
        c, _ = sess.run([loss, train], feed_dict = feed_dict)
        avg_loss += c/total_batch
    
    print(f'Epoch {epoch} \t===========>\t loss : {avg_loss:.8f}')

print('훈련 끝')

prediction = tf.equal(tf.compat.v1.argmax(hypothesis, 1), tf.compat.v1.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('accuracy : ', sess.run(accuracy, feed_dict = {x:x_test, y:y_test}))
#accuracy :  0.6171
import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

#1. 데이터
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)  #60000/100

x = tf.placeholder('float',[None,28,28,1])
y = tf.placeholder('float',[None,10])

#2. 모델구성

#l1. 
w1 = tf.get_variable("w1",shape=[3,3,1,32]) #3,3,1,32  3,3 => 커널사이즈 1 => 채널 32=> filtters(output)
L1 = tf.nn.conv2d(x,w1,strides=[1,1,1,1], padding = 'SAME') #양사이드의 1은 쉐입 맞춰주기 위함 
print(L1)
#conv2d(fillter, kernel_size, input_shape) summary??
#conv2d(32,(3,3),input_shape=(28,28,1))  파라미터의 갯수는? 
#(커널사이즈 x 채널 + 바이어스) x 필터

L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#maxpooling에서 padding은 큰의미없음
print(L1)

w2 = tf.get_variable("w2",shape=[3,3,32,64]) #3,3,1,32  3,3 => 커널사이즈 1 => 채널 32=> filtters(output)
L2 = tf.nn.conv2d(L1,w2,strides=[1,1,1,1], padding = 'SAME') #양사이드의 1은 쉐입 맞춰주기 위함 
L2 = tf.nn.selu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#maxpooling에서 padding은 큰의미없음
print(L2)

w3 = tf.get_variable("w3",shape=[3,3,64,128]) #3,3,1,32  3,3 => 커널사이즈 1 => 채널 32=> filtters(output)
L3 = tf.nn.conv2d(L2,w3,strides=[1,1,1,1], padding = 'SAME') #(None,7,7,128) 
L3 = tf.nn.selu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L3)

w4 = tf.get_variable("w4",shape=[3,3,128,64]) #3,3,1,32  3,3 => 커널사이즈 1 => 채널 32=> filtters(output)
L4 = tf.nn.conv2d(L3,w4,strides=[1,1,1,1], padding = 'SAME') #(None,7,7,128) 
L4 = tf.nn.selu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L4)  #(?,2,2,64)

# Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)->L1
# Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)->L2
# Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)->L3
# Tensor("MaxPool_3:0", shape=(?, 2, 2, 64), dtype=float32)->L4

# Flatten
L_flat = tf.reshape(L4,[-1,2*2*64])
print(L_flat)
# Tensor("Reshape:0", shape=(?, 256), dtype=float32)

#L5.
w5 = tf.get_variable("W5", shape=[2*2*64,64], initializer=tf.contrib.layers.variance_scaling_initializer())
b5 = tf.Variable(tf.random_normal([64],name='b5'))
L5 = tf.nn.selu(tf.matmul(L_flat,w5)+b5)
#L5 = tf.nn.dropout(L5, keep_prob = 0.2)
print(L5)

#L6.
w6 = tf.get_variable("W6", shape=[64,32], initializer=tf.contrib.layers.variance_scaling_initializer()) #tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([32],name='b6'))
L6 = tf.nn.selu(tf.matmul(L5,w6)+b6)
#L6 = tf.nn.dropout(L6, keep_prob = 0.2)
print(L6)

#L7.
w7 = tf.get_variable("W7", shape=[32,10], initializer=tf.contrib.layers.variance_scaling_initializer())
b7 = tf.Variable(tf.random_normal([10],name='b7'))
hypothesis = tf.nn.softmax(tf.matmul(L6,w7)+b7)
print("최종출력 :",hypothesis)

#3. 컴파일 훈련
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))
train = tf.train.AdamOptimizer(learning_rate=0.00005).min.imize(loss)

#훈련 
sess = tf.Session()
sess.run(tf.global_variables_initializer())

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

prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('accuracy : ', sess.run(accuracy, feed_dict = {x:x_test, y:y_test}))

#learning_rate =0.0001
#xavier 초기화 ,dropout 0.1-> accuracy :  0.6997
#he 초기화  -> accuracy :  0.9883
#드랍아웃 안넣어주는게 더 잘나옴 

















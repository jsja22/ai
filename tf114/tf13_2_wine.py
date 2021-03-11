#accuracy score

import tensorflow as tf
import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score

tf.set_random_seed(66)
datasets = load_wine()

x_data = datasets.data
y_data = datasets.target
y_data = datasets.target.reshape(-1,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=66)

#one hot
sess = tf.Session()
y_train = tf.one_hot(y_train,depth=3).eval(session=sess)
y_train = y_train.reshape(-1,3)
print(x_train.shape, x_test.shape)         
print(y_train.shape, y_test.shape)

# (142, 13) (36, 13)
# (142, 3) (36, 1)

x = tf.placeholder('float', shape=[None,13])
y = tf.placeholder('float', shape=[None,3])

w = tf.Variable(tf.zeros([13,3]), name = 'weight')
b = tf.Variable(tf.zeros([1,3]), name = 'bias')

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

train = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(5001):
        _, cost_val= sess.run([train, loss], feed_dict={x:x_train, y:y_train})

        if step % 100 == 0:
            print(f'{step} cost_v : {cost_val:.5f}')

    y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
    y_pred = np.argmax(y_pred, axis= 1)
    print('accuracy_score : ', accuracy_score(y_test, y_pred)) #accuracy_score :  1.0
#accuracy_score :  0.9722222222222222

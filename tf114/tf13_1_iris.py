#accuracy score

import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

tf.set_random_seed(66)
datasets = load_iris()

x_data = datasets.data
y_data = datasets.target
y_data = datasets.target.reshape(-1,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=66)

encoder = OneHotEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray()

print(x_train.shape, x_test.shape)         
print(y_train.shape, y_test.shape)

# (120, 4) (30, 4)
# (120, 3) (30, 1)

x = tf.placeholder('float', shape=[None,4])
y = tf.placeholder('float', shape=[None,3])

w = tf.Variable(tf.random_normal([4,3]), name='weight')
b = tf.Variable(tf.random_normal([3]), name='bias')

hypothesis =  tf.nn.softmax(tf.matmul(x, w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis),axis=1))

train = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(5001):
        _, cost_val = sess.run([train, loss], feed_dict={x:x_train, y:y_train})

        if step % 100 == 0:
            print(f'{step} cost_v : {cost_val:.5f}')

    y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
    y_pred = np.argmax(y_pred, axis= 1)
    print('accuracy_score : ', accuracy_score(y_test, y_pred)) #accuracy_score :  1.0
        
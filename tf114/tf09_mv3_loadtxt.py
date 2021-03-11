import tensorflow as tf
import numpy as np
tf.set_random_seed(66) 

dataset=np.loadtxt('C:/data/data-01-test-score.csv',delimiter=',', dtype = np.float32 )

x_data = dataset[:,0:-1]
y_data = dataset[:,[-1]]

x = tf.placeholder(tf.float32, shape = [None,3])
y = tf.placeholder(tf.float32, shape = [None,1])

w = tf.Variable(tf.random_normal([3,1]),name ='weight')
b = tf.Variable(tf.random_normal([1]),name ='bias')

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.8)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    _, cost_val, hy_val = sess.run([train, cost, hypothesis], 
    feed_dict = {x: x_data, y: y_data})

    if step % 10 ==0:
        print(step, "cost:", cost_val, "\n", hy_val)
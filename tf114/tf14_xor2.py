import tensorflow as tf
import numpy as np

tf.set_random_seed(66)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)
print(x_data.shape)
print(y_data.shape)

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

w1 = tf.Variable(tf.random_normal([2, 100]), name='weight1')    
b1 = tf.Variable(tf.random_normal([100]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)
#model.add(Dense(100, input_dim = 2, activation='sigmoid))

w2 = tf.Variable(tf.random_normal([100, 20]), name='weight2')        
b2 = tf.Variable(tf.random_normal([20]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)
#model.add(Dense(20,activation='sigmoid))

w3 = tf.Variable(tf.random_normal([20, 1]), name='weight3')        
b3 = tf.Variable(tf.random_normal([1]), name='bias3')
hypothesis = tf.sigmoid(tf.matmul(layer2, w3) + b3)
#model.add(Dense(1,activation='sigmoid))

loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1-hypothesis))
train = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(4001):
        loss_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
        if step % 40 == 0:
            print(f'step : {step} \nloss : {loss_val} ')


    h , c, a = sess.run([hypothesis,predicted,accuracy], feed_dict={x:x_data, y:y_data})

    print(f'predict value : {h[0:5]} \n "original value: \n{c[0:5]} \naccuracy: : {a}')
    
#accuracy: : 1.0
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.set_random_seed(66)

dataset = load_boston()
x_data = dataset.data
y_data = dataset.target.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = 0.8, random_state= 66)

x = tf.placeholder(tf.float32, shape = [None, 13])
y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random.normal([13,1]), name = 'weight')
b = tf.Variable(tf.random.normal([1]), name = 'bias')

hypothesis = tf.matmul(x, w) + b

loss = tf.reduce_mean(tf.square(hypothesis - y))

train = tf.train.AdamOptimizer(learning_rate= 0.01).minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(40001):
        loss_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_train, y:y_train})
        if step % 40 == 0:
            print(f'step : {step} \nloss : {loss_val} ')
    y_pred = sess.run(hypothesis, feed_dict={x:x_test})
    print('R2 : ', r2_score(y_test, y_pred))
# loss : 23.692602157592773
# R2 :  0.8165261069817504
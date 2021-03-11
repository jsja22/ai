
#accuracy score
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import tensorflow as tf
tf.set_random_seed(66)

dataset = load_breast_cancer()

x_data = dataset.data
y_data = dataset.target
y_data = y_data.reshape(-1,1)
print(x_data.shape)
print(y_data.shape)
# (569, 30)
# (569, 1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

x= tf.placeholder(tf.float32, shape=[None,30])
y= tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.zeros([30,1]),name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x,w)+b)
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))  #binary_crossentropy

train = tf.train.AdamOptimizer(learning_rate=1e-7).minimize(loss)


predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32) 

#텐서를 새로운 형태로 캐스팅하는데 사용한다.
#부동소수점형에서 정수형으로 바꾼 경우 소수점 버린을 한다.
#Boolean형태인 경우 True이면 1, False이면 0을 출력한다.
#0.5 이상 1 0.5 보다 작으면 0

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype = tf.float32))
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(40001):
        loss_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_train, y:y_train})
        if step % 100 == 0:
            print(f'step : {step} \nloss : {loss_val} ')


    h , c, a = sess.run([hypothesis,predicted,accuracy], feed_dict={x:x_test, y:y_test})

    print(f'predict value : {h[0:5]} \n "original value: \n{c[0:5]} \naccuracy: : {a}')

    y_pred = sess.run(hypothesis, feed_dict={x:x_test})
    y_pred = np.where(y_pred>0.5, 1, 0)
    print('accuracy_score : ', accuracy_score(y_test, y_pred))

#accuracy: : 0.9210526347160339

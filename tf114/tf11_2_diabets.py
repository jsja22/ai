#r2_score

from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import tensorflow as tf
tf.set_random_seed(66)
dataset = load_diabetes()

x_data = dataset.data
y_data = dataset.target
y_data = y_data.reshape(-1,1)
print(x_data.shape)
print(y_data.shape)
# (442, 10)
# (442,)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = 0.8)

x= tf.placeholder(tf.float32, shape=[None,10])
y= tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([10,1]),name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis = tf.sigmoid(tf.matmul(x,w)+b)
# cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))  #binary_crossentropy

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-10)
# train = optimizer.minimize(cost)

# predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32) 

# #텐서를 새로운 형태로 캐스팅하는데 사용한다.
# #부동소수점형에서 정수형으로 바꾼 경우 소수점 버린을 한다.
# #Boolean형태인 경우 True이면 1, False이면 0을 출력한다.
# #0.5 이상 1 0.5 보다 작으면 0

# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype = tf.float32))
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())

#     for step in range(5001):
#         cost_val, _ = sess.run([cost,train], feed_dict={x:x_data, y:y_data})

#         if step % 200 ==0:
#             print(step,cost_val)

#     h , c, a = sess.run([hypothesis,predicted,accuracy], feed_dict={x:x_data, y:y_data})

#     print("predict value :", h, "original value: ", c, "accuracy: ", a)


hypothesis = tf.matmul(x, w) + b

loss = tf.reduce_mean(tf.square(hypothesis - y))

train = tf.train.AdamOptimizer(learning_rate= 5*1e-1).minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(10001):
        loss_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_train, y:y_train})
        if step % 40 == 0:
            print(f'step : {step} \nloss : {loss_val} ')
    y_pred = sess.run(hypothesis, feed_dict={x:x_test})
    print('R2 : ', r2_score(y_test, y_pred))
    
# loss : 2930.112060546875
# R2 :  0.5390982141925973
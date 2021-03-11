import tensorflow as tf
tf.set_random_seed(66)

x_train = tf.placeholder(tf.float32,shape=[None])
y_train = tf.placeholder(tf.float32,shape=[None])

W=tf.Variable(tf.random_normal([1]),name='weight')  #정규분포에따른 랜덤값을 하나 넣겠다는 뜻
b=tf.Variable(tf.random_normal([1]),name='bias')

sess = tf.Session()

hypothesis = x_train * W +b

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) #예측값에서 실제값을 뺸것을 제곱을하여 평균을 낸것 -> 비용(손실) => loss=mse 와 같다 !

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost) #AdamOptimizer 성능 최고

#train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    for step in range(2000):  #step -> epoch
        W_val , _, cost_val, b_val = sess.run([W,train, cost, b], feed_dict={x_train:[1,2,3], y_train:[3,5,7]})

        if step % 20 ==0:
            
            print("step: ",step,'cost_val: ',cost_val,'W_val: ',W_val,'b_val: ',b_val)

    print("예측1: ", sess.run(hypothesis, feed_dict = {x_train:[4]})) 
    print("예측2: ", sess.run(hypothesis, feed_dict = {x_train:[5, 6]}))        
    print("예측3: ", sess.run(hypothesis, feed_dict = {x_train:[6, 7, 8]}))  
# 예측1:  [8.998173]
# 예측2:  [10.997114 12.996057]
# 예측3:  [12.996057 14.994999 16.99394 ]
    
#경사하강법에있는 최적의 optimizer (minimize)해준 지점을 찾음
#loss가 최소인것을 찾음
#1. x*w + b  =>와 y_train에서 mse를 산출
#2. 계산된 cost를 minimized해준것이 optimizer
#3. optimizer를 train  1,2,3 한번 돈것이 1 epoch 
sess.close()

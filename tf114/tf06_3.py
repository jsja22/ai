import tensorflow as tf
tf.set_random_seed(66)

x_train = tf.placeholder(tf.float32,shape=[None])
y_train = tf.placeholder(tf.float32,shape=[None])

W=tf.Variable(tf.random_normal([1]),name='weight')  #정규분포에따른 랜덤값을 하나 넣겠다는 뜻
b=tf.Variable(tf.random_normal([1]),name='bias')

sess = tf.Session()

hypothesis = x_train * W +b

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) #예측값에서 실제값을 뺸것을 제곱을하여 평균을 낸것 -> 비용(손실) => loss=mse 와 같다 !

train = tf.train.AdamOptimizer(learning_rate=0.41).minimize(cost) #AdamOptimizer 성능 최고
train = tf.train.GradientDescentOptimizer(learning_rate=0.1741).minimize(cost)
#러닝레이트가 높을수록 경사를 빠르게 내려가니까 lr 을 높여 빨리 내려간다 - 에폭을 줄일수있다
#train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    for step in range(101):  #step -> epoch
        W_val , _, cost_val, b_val = sess.run([W,train, cost, b], feed_dict={x_train:[1,2,3], y_train:[3,5,7]})

        if step % 20 ==0:
            
            print("step: ",step,'cost_val: ',cost_val,'W_val: ',W_val,'b_val: ',b_val)

    print("예측1: ", sess.run(hypothesis, feed_dict = {x_train:[4]})) 
    print("예측2: ", sess.run(hypothesis, feed_dict = {x_train:[5, 6]}))        
    print("예측3: ", sess.run(hypothesis, feed_dict = {x_train:[6, 7, 8]}))  

#train = tf.train.GradientDescentOptimizer(learning_rate=0.1741).minimize(cost)

# step:  0 cost_val:  31.947226 W_val:  [3.723019] b_val:  [2.097327]
# step:  20 cost_val:  0.034452684 W_val:  [2.0030916] b_val:  [1.1516579]
# step:  40 cost_val:  0.0005160976 W_val:  [1.9773173] b_val:  [1.0565852]
# step:  60 cost_val:  9.497681e-05 W_val:  [1.989193] b_val:  [1.0247254]
# step:  80 cost_val:  1.8621888e-05 W_val:  [1.9951897] b_val:  [1.0109396]


#train = tf.train.AdamOptimizer(learning_rate=0.41).minimize(cost) #AdamOptimizer 성능 최고
#
# step:  0 cost_val:  31.947226 W_val:  [0.08333015] b_val:  [0.73996735]
# step:  20 cost_val:  1.3527976 W_val:  [1.2074965] b_val:  [1.4715265]
# step:  40 cost_val:  0.038589258 W_val:  [1.7330803] b_val:  [1.4550314]
# step:  60 cost_val:  0.011183954 W_val:  [1.9043503] b_val:  [1.2148714]
# step:  80 cost_val:  0.0006710254 W_val:  [1.9679132] b_val:  [1.0553634]
# step:  100 cost_val:  0.00018420193 W_val:  [1.9922658] b_val:  [0.9974681]
# 예측1:  [8.996167]
# 예측2:  [10.993947 12.991728]
# 예측3:  [12.991728 14.989509 16.98729 ]
sess.close()

import tensorflow as tf
tf.set_random_seed(66)

x_train =[1,2,3]
y_train = [3,5,7]

W=tf.Variable(tf.random_normal([1]),name='weight')  #정규분포에따른 랜덤값을 하나 넣겠다는 뜻
b=tf.Variable(tf.random_normal([1]),name='bias')

hypothesis = x_train * W +b

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) #예측값에서 실제값을 뺸것을 제곱을하여 평균을 낸것 -> 비용(손실) => loss=mse 와 같다 !

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) #AdamOptimizer 성능 최고

train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(W),sess.run(b)) #[0.06524777] [1.4264158]

for step in range(3):  #step -> epoch
    sess.run(train)
    if step % 1 ==0:
        print("step: ",step,'sess.run(cost): ',sess.run(cost),'sess.run(W): ',sess.run(W),'sess.run(b): ',sess.run(b)) #weight 2, bias 1 로 수렴
    # sess.run(train)
    # if step % 2 ==0:
    #     print("step: ",step,'sess.run(cost): ',sess.run(cost),'sess.run(W): ',sess.run(W),'sess.run(b): ',sess.run(b)) #weight 2, bias 1 로 수렴


#경사하강법에있는 최적의 optimizer (minimize)해준 지점을 찾음
#loss가 최소인것을 찾음
#1. x*w + b  =>와 y_train에서 mse를 산출
#2. 계산된 cost를 minimized해준것이 optimizer
#3. optimizer를 train  1,2,3 한번 돈것이 1 epoch 
import tensorflow as tf
tf.compat.v1.set_random_seed(66)

W=tf.Variable(tf.compat.v1.random_normal([1]),name='weight')  #정규분포에따른 랜덤값을 하나 넣겠다는 뜻
#b=tf.Variable(tf.random_normal([1]),name='bias')

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

aaa = sess.run(W)
print("aaa:",aaa)
sess.close()

#InteractiveSession(): 위와 결과는 같지만 변수.eval을 써줘야
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = W.eval()
#변수.eval
print("bbb:",bbb)
sess.close()

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session=sess)  #session을 명시해줘야함
print("ccc:",ccc)
sess.close()


# aaa: [0.06524777]
# bbb: [0.06524777]
# ccc: [0.06524777]
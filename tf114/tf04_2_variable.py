import tensorflow as tf

sess = tf.Session()

x = tf.Variable([2],dtype=tf.float32,name='test')

init = tf.compat.v1.global_variables_initializer()

#변수사용할때 위와같이 초기화 해주어야한다(텐서플로우에 쓸 수 있게 초기화 해주는것)

sess.run(init)
print(sess.run(x))  #[2.]
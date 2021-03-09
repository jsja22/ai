import tensorflow as tf

print(tf.__version__) #1.14.0

hello = tf.constant("Hello World") #Tensor("Const:0", shape=(), dtype=string)
print(hello)

sess = tf.Session()   #세션을 만들어주어야 문자형을 출력할 수 있다. 
print(sess.run(hello))


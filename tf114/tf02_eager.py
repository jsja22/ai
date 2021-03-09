#즉시 실행 모드
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf

print(tf.executing_eagerly()) #tf114#False  #base#True #즉시 실행모드 -> sess run이 없어진것 (텐서 2에서 사라짐)

tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly()) #False

print(tf.__version__) #1.14.0

hello = tf.constant("Hello World") #Tensor("Const:0", shape=(), dtype=string)
print(hello)

sess = tf.compat.v1.Session()   #세션을 만들어주어야 문자형을 출력할 수 있다. 
print(sess.run(hello))

#'tensorflow' has no attribute 'Session'  텐서 2에서는 세션이 존재하지 않다고 오류뜸 
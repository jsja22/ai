#실습
#덧셈 뺄샘 곱셈 나누셈 
import tensorflow as tf
def sRun(input): 
    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer()) 
        rst = input.eval() 
        print(rst) 
        return(rst)

c1 = tf.constant([[1, 2], [3, 4]])  #2x2 행렬
c2 = tf.constant([[4, 7], [-1, 0]]) 
e = tf.constant([[1, 0], [0, 1]]) 

print(sRun(tf.add(c1, c2)))
print(sRun(tf.subtract(c1, c2)))
# [[-3 -5]
#  [ 4  4]]
# [[-3 -5]
#  [ 4  4]]
print(sRun(tf.multiply(c1,e))) #원소곱
print(sRun(tf.matmul(c1,e))) #행렬곱
print(sRun(tf.divide(c1,c2)))  #나눗셈
#print(sRun(tf.mod(c1,c2))) #나눈 몫 
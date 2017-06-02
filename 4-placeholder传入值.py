
import tensorflow as tf

input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)

output=tf.mul(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))

#如果用了placeholder，就意味着你要在sess.run的时候传入新的输入用feed_dict
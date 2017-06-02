
import tensorflow as tf


def add_layer(inputs,in_size,out_size,activation_function=None):
    #定义权重为随机矩阵
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    #biases推荐初始值不为0
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs爬虫大佬








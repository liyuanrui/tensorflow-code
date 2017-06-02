import tensorflow as tf
import numpy as np


#创建数据
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3


#创建tensorflow结构------------------

#参数变量，随机数列生成，1维结构，范围-1到1
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
#初始值0
biases = tf.Variable(tf.zeros([1]))
#线性方程
y = Weights*x_data + biases

#最小方差，预测值与真实值的误差
loss=tf.reduce_mean(tf.square(y-y_data))
#optimizer优化器减少误差
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

#初始化参数变量
init = tf.initialize_all_variables()
#------------------------------------

#激活结构，非常重要
sess = tf.Session()
sess.run(init)

#训练
for step in range(201):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(Weights),sess.run(biases))


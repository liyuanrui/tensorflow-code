
import tensorflow as tf

#定义参数变量
state=tf.Variable(0,name='counter')

# print(state.name)

#一个常量
one=tf.constant(1)
#加法
new_value=tf.add(state,one)
#update参数变量state
update=tf.assign(state,new_value)
#初始化，如果有定义参数变量
init=tf.initialize_all_variables()

with tf.Session() as sess:
    #run init
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))



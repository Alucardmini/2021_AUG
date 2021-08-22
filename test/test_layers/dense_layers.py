# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2021/8/8 1:35 PM'


import tensorflow as tf
import numpy as np

arr = np.random.random((64, 100, 8))

att_size = 64

x = tf.placeholder(tf.float32, shape=[64, 100, 8], name="x")
u_context = tf.Variable(tf.truncated_normal([64]))

h = tf.layers.dense(x, units=64, activation=tf.tanh)

hu_sum = tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True)
exp = tf.exp(hu_sum)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)


    h_val, u_context_val, hu_sum_val= sess.run([h, u_context, hu_sum], feed_dict={x:arr})

    print(h_val.shape)
    print(u_context_val.shape)
    print(hu_sum_val.shape)



# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2021/8/10 9:28 PM'


import tensorflow as tf

a = tf.random_normal(shape=[2, 2], mean=0, stddev=0.1)
b = tf.random_normal(shape=[2, 3], mean=0, stddev=0.1)


with tf.Session() as sess:
    print(sess.run(tf.tensordot(a, b, axes=1)).shape)
    # print(sess.run(tf.tensordot(a, b, axes=2)).shape)
    # print(sess.run(tf.tensordot(a, b, axes=3)).shape)
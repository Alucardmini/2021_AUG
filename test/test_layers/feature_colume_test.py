# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2021/8/8 2:25 PM'


import tensorflow as tf
sess=tf.Session()

features = {
    "birthplace": [[1], [1], [3], [4], [2], [3]]
}

birthplace = tf.feature_column.categorical_column_with_identity(key="birthplace", num_buckets=4, default_value=0)
birthplace = tf.feature_column.indicator_column(birthplace)

columnes = {
    birthplace
}

inputs = tf.feature_column.input_layer(features=features, feature_columns=columnes)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    v = sess.run(inputs)

    print(v)
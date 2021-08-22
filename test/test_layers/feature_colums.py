# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2021/8/8 3:20 PM'

import tensorflow as tf

features = {
    'sex': [1, 1, 2, 1, 1, 2],
    'department': ['sport', 'sport', 'sport', 'drawing', 'gardening', 'travelling'],
}

depart = tf.feature_column.categorical_column_with_vocabulary_list("department", ['sport','drawing','gardening','travelling'], dtype=tf.string)

sex = tf.feature_column.categorical_column_with_identity(key='sex', num_buckets=3, default_value=0)

sex_depart = tf.feature_column.crossed_column([depart, sex], 16)

sex_depart = tf.feature_column.indicator_column(sex_depart)

columnes = [
    sex_depart
]

inputs = tf.feature_column.input_layer(features, columnes)
init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(tf.tables_initializer())
    sess.run(init)

    v = sess.run(inputs)
    print(v)
# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2021/8/9 9:28 PM'

import tensorflow as tf


features = {
    'sale': [1.2, 2.3, 1.2, 1.5, 2.2]
}

sale = tf.feature_column.numeric_column("sale", default_value=0.0)

columns = [
    sale
]

inputs = tf.feature_column.input_layer(features=features, feature_columns=sale)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    v = sess.run(inputs)

    print(v)
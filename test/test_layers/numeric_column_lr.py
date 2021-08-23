# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2021/8/9 9:28 PM'

import tensorflow as tf


# features = {
#     'sale': [1.2, 2.3, 1.2, 1.5, 2.2]
# }
#
# sale = tf.feature_column.numeric_column("sale", default_value=0.0)
#
# columns = [
#     sale
# ]

features = {
    'sex': ['male', 'female', 'male']
}

sex = tf.feature_column.categorical_column_with_hash_bucket('sex', 2, dtype=tf.string)
sex = tf.feature_column.indicator_column(sex)
user_sex = tf.feature_column.categorical_column_with_identity(key="sex", num_buckets=3, default_value=0)

columns = [
    sex
]


inputs = tf.feature_column.input_layer(features=features, feature_columns=columns)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    v = sess.run(inputs)

    print(v)
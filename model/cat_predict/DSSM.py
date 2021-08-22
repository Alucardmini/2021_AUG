# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2021/8/7 5:06 PM'

import tensorflow as tf
from tensorflow.python.ops import array_ops


def variable_summaries(var, name):
    # tf.feature_column.embedding
    with tf.name_scope("summaries"):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/'+ name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)


def build_deep_layers(input_x, param, name):
    with tf.variable_scope("{}_nn".format(name)):
        deep_emb = input_x
        for index, elem in enumerate(param['hidden_units']):
            deep_emb = tf.layers.dense(deep_emb, elem, activation=tf.nn.relu)
        deep_emb = tf.layers.dense(deep_emb, 64, activation=tf.nn.relu, name="{}_emb".format(name))

    return deep_emb


def embed_concat(embed_list):
    with tf.name_scope("embedding_concat"):
        emb_cat = tf.concat(embed_list, 1)
    return emb_cat


def cosine_score(query_norm, query_emb, target_emb):

    with tf.variable_scope("cos", reuse=tf.AUTO_REUSE):
        target_norm = tf.norm(target_emb, axis=-1)
        norm = tf.multiply(query_norm, target_norm)

        cosine_score_ = tf.matmul(query_emb, target_emb, transpose_b=True)

        consine = cosine_score_ / tf.expand_dims(norm, 1)

        consine = tf.squeeze(consine)

    return consine


def build_fm(input_x, embed_size, fea_size):
    with tf.variable_scope("second_factor"):
        second_factor = tf.reshape(input_x, shape=[-1, fea_size, embed_size])
        sum_square = tf.square(tf.reduce_sum(second_factor, 2))
        square_sum = tf.reduce_sum(tf.square(second_factor), 2)
        logit = 0.5 * tf.reduce_sum((sum_square - square_sum), -1, keep_dims=True)

    return logit


def attention_layer(inputs, train_weight_reshape, attention_size):

    with tf.name_scope("attention_layer"):

        u_context = tf.Variable(tf.truncated_normal([attention_size]))

        h = tf.layers.dense(inputs=inputs, units=attention_size, activation=tf.nn.tanh)

        hu_sum = tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True)

        exp = tf.exp(hu_sum)







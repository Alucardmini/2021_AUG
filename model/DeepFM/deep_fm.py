# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2021/9/4 1:08 PM'

from data.titanic_process import *

train_files = "/Users/wuxikun/Documents/metis/data/titanic_train.tfrecords"
feats = {item: tf.io.FixedLenFeature(shape=(1,), dtype=tf.string) for item in identity_feas}
feats = merge(feats,
              {item: tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64) for item in
               category_feas})
feats = merge(feats,
              {item: tf.io.FixedLenFeature(shape=(1,), dtype=tf.float32) for item
               in dense_feas})
feats = merge(feats,
              {item: tf.io.FixedLenFeature(shape=(1,), dtype=tf.string) for
               item in embed_feas})
label = {
    'label': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64)
}

feature_schema = merge(label, feats)


def read(feature_schema):


    input_files = ["/Users/wuxikun/Documents/metis/data/titanic_train.tfrecords"]

    dataset = tf.data.TFRecordDataset(input_files)

    def parser(record):
        features = tf.parse_single_example(
            record,
            features=feature_schema
        )
        return features["label"], features['Sex']

    dataset = dataset.map(parser)  # 接受的参数是一个函数

    iterator = dataset.make_initializable_iterator()
    return iterator, iterator.get_next()


if __name__ == '__main__':
    iterator, next_elem = read(feature_schema)

    with tf.Session() as sess:

        sess.run(iterator.initializer)

        for i in range(10):
            sex = tf.decode_raw(next_elem[1], tf.uint8)
            print(sess.run(next_elem[0]))
            print(sess.run(sex))




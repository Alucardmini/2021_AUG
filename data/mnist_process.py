# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2021/8/22 3:29 PM'
import tensorflow as tf
from keras.datasets import mnist


def write_data_sets(features, labels, path, filename='train.tfrecords'):
    tf.python_io.TFRecordWriter(path+"/.tmp")

    write = tf.python_io.TFRecordWriter(filename)

    for i in range(len(features)):
        example = tf.train.Example(features=tf.train.Features(

            feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[features[i].tostring() ]))
            }

        ))

        write.write(example.SerializeToString())

    write.close()


def build_mnist_datasets(path, datasets="mnist"):

    if datasets == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        write_data_sets(x_train, y_train, path, 'train.tfrecords')
        write_data_sets(x_test, y_test, path, 'test.tfrecords')


if __name__ == '__main__':

    build_mnist_datasets("/Users/wuxikun/Documents/metis/data")









# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2021/9/4 1:08 PM'

from data.titanic_process import *


def get_feature_schema():

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
    return feature_schema


def read():

    input_files = ["/Users/wuxikun/Documents/metis/data/titanic_train.tfrecords"]

    dataset = tf.data.TFRecordDataset(input_files)

    def parser(record):
        features = tf.parse_single_example(
            record,
            features=get_feature_schema()
        )


        return features["label"], features['Sex'], features['Embarked'], features['Pclass'], features['Parch'],features['Age'],features['SibSp'],features['Fare'],features['Ticket'],features['Cabin']

    dataset = dataset.map(parser)  # 接受的参数是一个函数

    iterator = dataset.make_initializable_iterator()
    return iterator, iterator.get_next()


def get_feature_columns():

    sex = tf.feature_column.categorical_column_with_hash_bucket('sex', hash_bucket_size=3, dtype=tf.string)
    sex = tf.feature_column.indicator_column(sex)

    embarked = tf.feature_column.categorical_column_with_hash_bucket('Embarked', hash_bucket_size=3, dtype=tf.string)
    embarked = tf.feature_column.indicator_column(embarked)

    pclass = tf.feature_column.categorical_column_with_identity("Pclass", num_buckets=10, default_value=0)
    pclass = tf.feature_column.indicator_column(pclass)
    age = tf.feature_column.numeric_column("Age", default_value=0.0)

    columns = [
        sex, embarked, pclass, age
    ]

    return columns




def model_fn(features, labels, mode, params):

    feature_columns = params['feature_column']

    input_layers = tf.feature_column.input_layer(features=features, feature_columns=feature_columns)

    fc1 = tf.layers.dense(input_layers, 1024)
    fc1 = tf.layers.dropout(fc1, rate=0.4)
    y_pred = tf.layers.dense(fc1, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"result": tf.arg_max(y_pred, 1)})

    loss = tf.reduce_mean(tf.square(y_pred - labels))
    train_op = tf.train.GradientDescentOptimizer(learning_rate=params['lr_rate']).minimize(loss)

    return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss=loss)



if __name__ == '__main__':
    # iterator, next_elem = read()

    model_params = {"lr_rate": 0.01,
                    "feature_column": get_feature_columns()
                    }
    model = tf.estimator.Estimator(model_fn=model_fn, params=model_params, model_dir="model/lr")


    train_spec = tf.estimator.TrainSpec(read)


    # iterator, next_elem = read(get_feature_schema())
    #
    # with tf.Session() as sess:
    #
    #     sess.run(iterator.initializer)
    #
    #     for i in range(10):
    #
    #         print(sess.run(next_elem))








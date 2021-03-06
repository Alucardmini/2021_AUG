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


        # return features["label"], features['Sex'], features['Embarked'], features['Pclass'], features['Parch'],features['Age'],features['SibSp'],features['Fare'],features['Ticket'],features['Cabin']
        return features

    dataset = dataset.map(parser)  # 接受的参数是一个函数

    dataset = dataset.repeat(10).shuffle(64).batch(32)

    iterator = dataset.make_one_shot_iterator()
    next_elem = iterator.get_next()
    # labels = next_elem[0]
    # return next_elem[1:], labels
    # return iterator, next_elem
    return next_elem, next_elem.pop("label")


def get_feature_columns():

    sex = tf.feature_column.categorical_column_with_hash_bucket('Sex', hash_bucket_size=3, dtype=tf.string)
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

    input_layers = tf.feature_column.input_layer(features=features, feature_columns=get_feature_columns())
    fc1 = tf.layers.dense(input_layers, 1024)
    fc1 = tf.layers.dropout(fc1, rate=0.4)
    y_pred = tf.layers.dense(fc1, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"result": tf.arg_max(y_pred, 1)})

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=tf.cast(labels, tf.float32)))

    train_op = tf.train.GradientDescentOptimizer(learning_rate=params['lr_rate']).minimize(loss)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(tf.arg_max(y_pred, 1), labels)}

    return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss=loss, eval_metric_ops=eval_metric_ops)


def serving_input_fn():
    # features_placeholder = tf.placeholder(tf.float32, [None])
    # labels_placeholder = tf.placeholder(tf.float32, [None])

    inputs = {'Sex': tf.placeholder(tf.string, [None], name="Sex"),
              'Embarked': tf.placeholder(tf.string, [None], name="Embarked"),
              'Pclass': tf.placeholder(tf.int32, [None], name="Pclass"),
              'Age': tf.placeholder(tf.float32, [None], name="Age")
              }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


if __name__ == '__main__':
    # iterator, next_elem = read()

    model_params = {"lr_rate": 0.01,
                    "feature_column": get_feature_columns()
                    }
    model = tf.estimator.Estimator(model_fn=model_fn, params=model_params, model_dir="model/lr")

    iterator, next_elem = read()

    train_spec = tf.estimator.TrainSpec(input_fn=read, max_steps=20000)
    eval_spec = tf.estimator.EvalSpec(read)

    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

    model.export_saved_model("model/lr/saved_model", serving_input_fn)










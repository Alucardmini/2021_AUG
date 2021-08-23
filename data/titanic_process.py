# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2021/8/22 5:55 PM'
import  tensorflow as tf

import csv

schmes = ["PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]

label = 'Survived'

identity_feas = ["Sex", "Embarked"]
category_feas = ["Pclass", "Parch"]
dense_feas = ["Age", "SibSp", "Fare"]
embed_feas = ["Ticket", "Cabin"]


def build_titanic_datasets(input_path, output_path, output_name):
    tf.python_io.TFRecordWriter(output_path + "/.tmp")

    write = tf.python_io.TFRecordWriter(output_name)

    with open(input_path, 'r') as f:

        reader = csv.reader(f)
        succ_cnt = 0
        fail_cnt = 0
        for elems in reader:
            try:

                feats = {item: tf.train.Feature(bytes_list=tf.train.BytesList(value=[elems[schmes.index(item)].encode()])) for item in identity_feas}
                feats = merge(feats, {item: tf.train.Feature(int64_list=tf.train.Int64List(value=[int(elems[schmes.index(item)])])) for item in category_feas})
                feats = merge(feats, {item: tf.train.Feature(float_list=tf.train.FloatList(value=[float(elems[schmes.index(item)])])) for item in dense_feas})
                feats = merge(feats, {item: tf.train.Feature(bytes_list=tf.train.BytesList(value=[elems[schmes.index(item)].encode()])) for item in embed_feas})
                label = {
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(elems[1])]))
                }

                feats = merge(label, feats)

                example = tf.train.Example(features=tf.train.Features(
                    feature=feats
                ))

                write.write(example.SerializeToString())

                succ_cnt += 1


            except Exception as e:
                fail_cnt += 1

        write.close()

        print("succ_cnt===>", succ_cnt, " fail_cnt==>", fail_cnt)

def merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

if __name__ == "__main__":

    out_path="/Users/wuxikun/Documents/metis/data"

    build_titanic_datasets('./titanic_train.csv', out_path, 'titanic_train.tfrecords')




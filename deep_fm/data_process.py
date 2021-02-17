import pickle

import numpy as np
import tensorflow as tf
import pandavro as pdx

with open("column_idx.pkl", "rb") as f:
    COLUMNS = pickle.load(f)


def _create_tfrecord(file):
    data = pdx.from_avro("TrainTemp/" + file)
    fname = file.replace("avro", "tfrecords")

    with tf.python_io.TFRecordWriter("TrainData/" + fname) as writer:
        for _, row in data.iterrows():
            index = [idx[row[col]] for col, idx in COLUMNS.items()]
            value = np.full(len(index), 1)
            label = row.click

            example = tf.train.Example()
            example.features.feature["index"].int64_list.value.extend(index)
            example.features.feature["value"].int64_list.value.extend(value)
            example.features.feature["label"].int64_list.value.append(label)
            writer.write(example.SerializeToString())

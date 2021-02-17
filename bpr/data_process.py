import random
import multiprocessing as mp
from string import ascii_letters, digits

import tensorflow as tf
import numpy as np


def garbled_generator(num):
    garbled = [random.choice(ascii_letters + digits) for _ in range(num)]
    return ''.join(garbled)


class NegSamplePool:
    def __init__(self, min_item, max_item):
        self.min_item = min_item
        self.max_item = max_item

    def _create_neg(self, positive):
        positive_num = positive.shape[0]
        candidates = np.random.choice(
            np.arange(self.min_item, self.max_item),
            positive_num * 2)

        negative = list()
        positive_set = set(positive)
        for i in candidates:
            if len(negative) == positive_num:
                negative = np.array(negative)
                break
            if i not in positive_set:
                negative.append(i)
        return negative

    def run(self, row):
        uid, arr = row
        positive_arr = np.array(arr)
        negative_arr = self._create_neg(positive_arr)
        user_arr = np.repeat(uid, positive_arr.shape[0])

        return np.hstack((
            user_arr[:, None],
            positive_arr[:, None],
            negative_arr[:, None]))


class TFRecordPool:
    def __init__(self, random_num):
        self.random_num = random_num

    def run(self, data):
        fname = "TrainData/train_{}.tfrecords".format(garbled_generator(self.random_num))
        with tf.python_io.TFRecordWriter(fname) as writer:
            for row in data:
                example = tf.train.Example()
                example.features.feature["user_id"].int64_list.value.extend([row[0]])
                example.features.feature["pos_feat"].int64_list.value.extend(row[1])
                example.features.feature["neg_feat"].int64_list.value.extend(row[2])
                writer.write(example.SerializeToString())


def _create_training_data(data):
    neg_pool = NegSamplePool(1, 326002)
    pool = mp.Pool(10)
    pos_neg_pair = pool.map(neg_pool.run, data)
    pool.close()
    pool.join()

    pairs = np.vstack(pos_neg_pair)
    tf_pool = TFRecordPool(5)
    pool = mp.Pool(20)
    pool.map(tf_pool.run, np.array_split(pairs, 10))
    pool.close()
    pool.join()

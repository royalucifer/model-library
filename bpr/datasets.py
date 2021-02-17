import numpy as np
import tensorflow as tf


def seriving_input_fn(directory, user_id, batch_size=500000):
    pos_feat = np.genfromtxt(directory, delimiter=',', skip_header=1, dtype=int)[:, :-1].reshape([-1])
    user_id = np.repeat(user_id, pos_feat.shape[0])
    features = {
        'user_id': user_id,
        'pos_feat': pos_feat}
    dataset = tf.data.Dataset.from_tensor_slices(features)
    return dataset.batch(batch_size)


def training_input_fn(directory, batch_size, num_epochs):
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=directory,
        batch_size=batch_size,
        features={
            'user_id': tf.FixedLenFeature([], tf.int64),
            'pos_feat': tf.FixedLenFeature([], tf.int64),
            'neg_feat': tf.FixedLenFeature([], tf.int64)},
        reader=tf.data.TFRecordDataset,
        num_epochs=num_epochs,
        shuffle=True,
        shuffle_buffer_size=100,
        reader_args=['GZIP'])
    return dataset

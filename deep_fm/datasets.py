import tensorflow as tf


def input_fn(directory, num_epochs=1, batch_size=100):
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=directory,
        batch_size=batch_size,
        features={
            "index": tf.FixedLenFeature([128], tf.int64),
            "value": tf.FixedLenFeature([128], tf.int64),
            "label": tf.FixedLenFeature([], tf.int64)},
        reader=tf.data.TFRecordDataset,
        num_epochs=num_epochs,
        shuffle=True,
        shuffle_buffer_size=10000)
    element = dataset.make_one_shot_iterator().get_next()
    feature = {"index": element["index"], "value": element["value"]}
    return feature, element['label']

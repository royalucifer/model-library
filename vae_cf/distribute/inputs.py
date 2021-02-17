import tensorflow as tf


class Dataset:
    class _ToDense:
        def __call__(self, indices, values, shape):
            sparse_tensor = tf.sparse.SparseTensor(
                indices, values, shape)
            dense_tensor = tf.sparse.to_dense(
                sparse_tensor, validate_indices=False)
            return dense_tensor

    class _ToEvalDense:
        def __call__(self, feat_indices, feat_values,
                     label_indices, label_values, shape):
            feature_sparse_tensor = tf.sparse.SparseTensor(
                feat_indices, feat_values, shape)
            label_sparse_tensor = tf.sparse.SparseTensor(
                label_indices, label_values, shape)
            feature_dense_tensor = tf.sparse.to_dense(
                feature_sparse_tensor, validate_indices=False)
            label_dense_tensor = tf.sparse.to_dense(
                label_sparse_tensor, validate_indices=False)
            return feature_dense_tensor, label_dense_tensor


def get_train_data(data, batch_size, num_epochs=1, strategy=None):
    def _sparse_generator():
        for _ in range(num_epochs):
            for start_idx in range(0, n_users, batch_size):
                end_idx = min(start_idx + batch_size, n_users)
                coo_row = data[start_idx:end_idx].tocoo()

                indices = list(zip(coo_row.row, coo_row.col))
                values = coo_row.data
                shape = [batch_size, n_items]
                yield (indices, values, shape)

    n_users, n_items = data.shape
    dataset = tf.data.Dataset.from_generator(
        _sparse_generator,
        output_types=(tf.int64, tf.float32, tf.int64),
        output_shapes=(tf.TensorShape([None, None]), tf.TensorShape([None]), tf.TensorShape([None])))
    dataset = dataset.map(Dataset._ToDense(), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if strategy:
        dataset = strategy.experimental_distribute_dataset(dataset)
    return dataset


def get_eval_data(features, labels, batch_size, strategy=None):
    def _sparse_generator():
        for start_idx in range(0, n_users, batch_size):
            end_idx = min(start_idx + batch_size, n_users)

            feat_row = features[start_idx:end_idx].tocoo()
            label_row = labels[start_idx:end_idx].tocoo()

            feat_indices = list(zip(feat_row.row, feat_row.col))
            feat_values = feat_row.data
            label_indices = list(zip(label_row.row, label_row.col))
            label_values = label_row.data

            shape = [batch_size, n_items]
            yield (feat_indices, feat_values, label_indices, label_values, shape)

    n_users, n_items = features.shape
    dataset = tf.data.Dataset.from_generator(
        _sparse_generator,
        output_types=(tf.int64, tf.float32, tf.int64, tf.float32, tf.int64),
        output_shapes=(
            tf.TensorShape([None, None]), tf.TensorShape([None]),
            tf.TensorShape([None, None]), tf.TensorShape([None]),
            tf.TensorShape([None])))
    dataset = dataset.map(Dataset._ToEvalDense(), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if strategy:
        dataset = strategy.experimental_distribute_dataset(dataset)
    return dataset

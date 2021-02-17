import math

import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf

from pickle_obj import Obj


def _create_weights(n_layers, n_users, n_items, embed_dim, weight_size, random_seed=None):
    w_init = tf.contrib.layers.xavier_initializer(seed=random_seed)
    b_init = tf.truncated_normal_initializer(stddev=0.001, seed=random_seed)
    w_size_list = [embed_dim] + weight_size

    all_weights = dict()

    all_weights['user_embed'] = tf.get_variable(
        name='user_embed', shape=[n_users, embed_dim], initializer=w_init)
    all_weights['item_embed'] = tf.get_variable(
        name='item_embed', shape=[n_items, embed_dim], initializer=w_init)

    for k in range(n_layers):
        all_weights['W_gc_%d' % k] = tf.get_variable(
            name='W_gc_%d' % k, shape=[w_size_list[k], w_size_list[k + 1]], initializer=w_init)
        all_weights['b_gc_%d' % k] = tf.get_variable(
            name='b_gc_%d' % k, shape=[w_size_list[k + 1]], initializer=b_init)

        all_weights['W_bi_%d' % k] = tf.get_variable(
            name='W_bi_%d' % k, shape=[w_size_list[k], w_size_list[k + 1]], initializer=w_init)
        all_weights['b_bi_%d' % k] = tf.get_variable(
            name='b_bi_%d' % k, shape=[w_size_list[k + 1]], initializer=b_init)
    return all_weights


class NGCFEmbed:
    def __init__(self, n_users, n_items, n_layers, n_folds):
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.n_folds = n_folds

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = list(zip(coo.row, coo.col))
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        keep_tensor = keep_prob + tf.random_uniform([n_nonzero_elems])
        dropout_mask = tf.cast(tf.floor(keep_tensor), dtype=tf.bool)
        out = tf.sparse_retain(X, dropout_mask)
        return out * tf.div(1., keep_prob)

    def _split_A_hat_node(self, X, drop_prob, mode):
        A_fold_hat = []
        fold_len = (n_users + n_items) // self.n_folds

        for i in range(self.n_folds):
            start = i * fold_len
            if i == self.n_folds - 1:
                end = self.n_users + self.n_items
            else:
                end = (i + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            if mode == tf.estimator.ModeKeys.TRAIN:
                n_nonzero_temp = X[start:end].count_nonzero()
                temp = self._dropout_sparse(temp, 1 - drop_prob, n_nonzero_temp)
            A_fold_hat.append(temp)

        return A_fold_hat

    def get(self, norm_adj, weights, node_drop_prob, mess_drop_prob, mode):
        A_fold_hat = self._split_A_hat_node(norm_adj, node_drop_prob, mode)
        ego_embed = tf.concat([weights['user_embed'], weights['item_embed']], axis=0)

        all_embed = [ego_embed]
        for k in range(self.n_layers):
            temp_embed = []
            for f in range(self.n_folds):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embed))

            # lambda * E
            side_embed = tf.concat(temp_embed, axis=0)
            # lambda * E * W_1
            sum_embed = tf.nn.leaky_relu(
                tf.matmul(side_embed, weights['W_gc_%d' % k]) + weights['b_gc_%d' % k])
            # lambda * E X E
            bi_embed = tf.multiply(ego_embed, side_embed)
            # lambda * E X E * W_2
            bi_embed = tf.nn.leaky_relu(
                tf.matmul(bi_embed, weights['W_bi_%d' % k]) + weights['b_bi_%d' % k])

            # E_next = (lambda * E_orig * W_1) + (lambda * E_orig X E_orig * W_2)
            ego_embed = sum_embed + bi_embed
            ego_embed = tf.nn.dropout(ego_embed, 1 - mess_drop_prob[k])
            norm_embed = tf.math.l2_normalize(ego_embed, axis=1)

            all_embed += [norm_embed]

        all_embed = tf.concat(all_embed, 1)
        u_g_embed, i_g_embed = tf.split(all_embed, [self.n_users, self.n_items], axis=0)
        return u_g_embed, i_g_embed


def _compute_ndcg(labels, logits, features, topn=100):
    zero = tf.constant(0, dtype=tf.float32)
    neg_val = tf.constant(-1000, dtype=tf.float32)

    condition = tf.not_equal(features, zero)
    predictions = tf.where_v2(condition, neg_val, logits)
    valid_list, _ = Metrics.compute_ndcg(labels, predictions, topn)
    return tf.metrics.mean(valid_list)


def model_fn(features, labels, mode, params):
    # --------------------- Hyperparameters -------------------- #
    norm_adj = params['norm_adj']
    n_users = params['n_users']
    n_items = params['n_items']
    n_folds = params['n_folds']
    n_layers = params['n_layers']
    embed_dim = params['embed_dim']
    weight_size = params['weight_size']

    l2_reg = params['l2_reg']
    batch_size = params['batch_size']
    node_drop_prob = params['node_drop_prob']
    mess_drop_prob = params['mess_drop_prob']
    random_seed = params['random_seed']
    learning_rate = params['learning_rate']

    # weights
    weights = _create_weights(n_layers, n_users, n_items, embed_dim, weight_size, random_seed)
    ngcf_embed = NGCFEmbed(n_users, n_items, n_layers, n_folds)

    # ------------------------ Calculate ----------------------- #
    user_embed, item_embed = ngcf_embed.get(norm_adj, weights, node_drop_prob, mess_drop_prob, mode)
    u_embed = tf.nn.embedding_lookup(user_embed, features['users'])

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = tf.matmul(u_embed, item_embed, transpose_a=False, transpose_b=True)
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    if mode == tf.estimator.ModeKeys.EVAL:
        labels = tf.sparse.to_dense(labels, validate_indices=False)
        logits = tf.matmul(u_embed, item_embed, transpose_a=False, transpose_b=True)
        loss = tf.constant(0.0, tf.float32, [1])
        eval_metric_ops = {
            "ndcg": _compute_ndcg(labels, logits, features)}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # Loss
    pos_i_embed = tf.nn.embedding_lookup(item_embed, features['pos_items'])
    neg_i_embed = tf.nn.embedding_lookup(item_embed, features['neg_items'])

    pos_scores = tf.reduce_sum(tf.multiply(u_embed, pos_i_embed), axis=1)
    neg_scores = tf.reduce_sum(tf.multiply(u_embed, neg_i_embed), axis=1)

    mf_loss = tf.reduce_mean(-tf.log_sigmoid(pos_scores - neg_scores))
    reg_loss = tf.nn.l2_loss(u_embed) + tf.nn.l2_loss(pos_i_embed) + tf.nn.l2_loss(neg_i_embed)
    reg_loss = 2 * l2_reg * (reg_loss / batch_size)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        loss = mf_loss + reg_loss

        opt = tf.train.AdamOptimizer(learning_rate)
        # opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
        train_op = opt.minimize(loss, global_step=global_step)

        tf.summary.scalar('mf_loss', mf_loss)
        tf.summary.scalar('loss', loss)

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)


class Dataset:
    class _ToDense:
        def __call__(self, indices, values, shape):
            sparse_tensor = tf.sparse.SparseTensor(
                indices, values, shape)
            return sparse_tensor

    class _ToEvalDense:
        def __call__(self, feat_indices, feat_values,
                     label_indices, label_values, shape):
            feat_sparse_tensor = tf.sparse.SparseTensor(
                feat_indices, feat_values, shape)
            label_sparse_tensor = tf.sparse.SparseTensor(
                label_indices, label_values, shape)
            return feat_sparse_tensor, label_sparse_tensor


def evaluate_input_fn(features, labels, batch_size):
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
    return dataset


def training_input_fn(data, num_epochs, batch_size, n_users):
    def _negative_sample(features):
        prob = [1 / n_users] * n_users
        dist = tf.distributions.Categorical(prob)
        features['neg_items'] = dist.sample(batch_size, seed=9876)
        return features

    n_users, n_items = data.shape
    dataset = tf.data.Dataset.from_tensor_slices(
        {"users": data['clientid'].values, "pos_items": data['pid'].values})
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(_negative_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset.repeat(num_epochs)


if __name__ == "__main__":
    PATH = '/home/jupyter/RecData/0728/'

    tr_data = pd.read_pickle(PATH + 'NGCF/pair_tr_data.pkl')
    norm_adj_mat = sp.sparse.load_npz(PATH + 'NGCF/norm_adj_mat.npz')
    vd_data_tr = sp.sparse.load_npz(PATH + 'NGCF/sparse_vd_data_tr.npz')
    vd_data_te = sp.sparse.load_npz(PATH + 'NGCF/sparse_vd_data_te.npz')
    vd_uid = Obj.load(PATH + 'NGCF/vd_uid.pkl')

    user_dict = Obj.load(PATH + 'NGCF/user_dict.pkl')
    item_dict = Obj.load(PATH + 'NGCF/item_dict.pkl')
    n_users = len(user_dict)
    n_items = len(item_dict)

    NUM_EPOCHS = 400
    TRAIN_BATCH_SIZE = 1024
    EVAL_BATCH_SIZE = 1024
    STEPS_PER_BATCH = math.ceil(n_users / TRAIN_BATCH_SIZE)

    model_params = {
        'norm_adj': norm_adj_mat,
        'n_users': n_users,
        'n_items': n_items,
        'n_folds': 100,
        'n_layers': 3,
        'embed_dim': 64,
        'weight_size': [64, 64, 64],
        'batch_size': TRAIN_BATCH_SIZE,
        'node_drop_prob': 0.1,
        'mess_drop_prob': [0.1,0.1,0.1],
        'random_seed': 98765,
        'learning_rate': 0.0001,
        'l2_reg': 1e-5}
    config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(
            device_count={'CPU': 8},
            gpu_options=tf.GPUOptions(allow_growth=True)),
        save_checkpoints_steps=10000,
        save_summary_steps=2000,
        log_step_count_steps=2000,
        keep_checkpoint_max=3)

    model_dir = "gs://udn-news-recserve/tensorboard/ngcf_test"
    NFCF = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=model_params, config=config)
    train_spec = NFCF.train(input_fn=lambda: training_input_fn(
        tr_data, num_epochs=NUM_EPOCHS, batch_size=TRAIN_BATCH_SIZE, n_users=n_users))

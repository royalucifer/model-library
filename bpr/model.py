import tensorflow as tf


def build_optimizer(name, learning_rate, loss, decay=False):
    global_step = tf.train.get_global_step()

    def _init_decay_lr(learning_rate):
        return tf.train.exponential_decay(
            learning_rate,
            global_step=global_step,
            decay_steps=100,
            decay_rate=0.5)

    if decay:
        learning_rate = _init_decay_lr(learning_rate)

    if name == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif name == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=1e-8)
    elif name == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.95)
    elif name == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    return optimizer.minimize(loss, global_step=global_step)


def model_fn(features, labels, mode, params):
    # --------------------- Hyperparameters -------------------- #
    embed_size = params['embed_size']
    user_feats_size = params['user_feats_size']
    item_feats_size = params['item_feats_size']

    l2_reg = params['l2_reg']
    learning_rate = params['learning_rate']
    optimizer_type = params['optimizer_type']

    # Weight
    user_v = tf.get_variable(
        name='user_v',
        shape=[user_feats_size, embed_size],
        initializer=tf.glorot_normal_initializer())
    item_v = tf.get_variable(
        name='item_v',
        shape=[item_feats_size, embed_size],
        initializer=tf.glorot_normal_initializer())
    user_b = tf.get_variable(
        name='user_b',
        shape=[user_feats_size],
        initializer=tf.glorot_uniform_initializer())
    item_b = tf.get_variable(
        name='item_b',
        shape=[item_feats_size],
        initializer=tf.glorot_uniform_initializer())

    # ------------------------ Calculate ----------------------- #
    # User
    with tf.variable_scope('User-Embedding'):
        user_feats = features['user_id']
        user_embed = tf.nn.embedding_lookup(user_v, user_feats)
        user_bias = tf.nn.embedding_lookup(user_b, user_feats)

    # Item - Positive
    with tf.variable_scope('Item-Embedding-Positive'):
        pos_feats = features['pos_feat']
        pos_embed = tf.nn.embedding_lookup(item_v, pos_feats)
        pos_bias = tf.nn.embedding_lookup(item_b, pos_feats)
        pos_score = tf.reduce_sum(user_embed * pos_embed, axis=1) + pos_bias

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = pos_score + user_bias
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    # Item - Negative
    with tf.variable_scope('Item-Embedding-Negative'):
        neg_feats = features['neg_feat']
        neg_embed = tf.nn.embedding_lookup(item_v, neg_feats)
        neg_bias = tf.nn.embedding_lookup(item_b, neg_feats)
        neg_score = tf.reduce_sum(user_embed * neg_embed, axis=1) + neg_bias

    # Loss
    with tf.variable_scope('Loss'):
        loss = tf.reduce_mean(-tf.log_sigmoid(pos_score - neg_score))
        train_op = build_optimizer(optimizer_type, learning_rate, loss, decay=False)

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)

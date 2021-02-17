import tensorflow as tf


def build_optimizer(name, learning_rate, loss):
    global_step = tf.train.get_global_step()

    def _init_decay_lr(learning_rate):
        return tf.train.exponential_decay(
            learning_rate,
            global_step=global_step,
            decay_steps=250,
            decay_rate=0.5)

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


def build_weight(feature_size, embed_size):
    weights = dict()
    weights['FM_V'] = tf.get_variable(
        name='fm_v',
        shape=[feature_size, embed_size],
        initializer=tf.glorot_normal_initializer()
    )
    weights['FM_W'] = tf.get_variable(
        name='fm_w',
        shape=[feature_size],
        initializer=tf.glorot_uniform_initializer()
    )
    weights['FM_B'] = tf.get_variable(
        name='fm_bias',
        shape=[1],
        initializer=tf.constant_initializer(0.0)
    )
    return weights


def model_fn(features, labels, mode, params):
    # --------------------- Hyperparameters -------------------- #
    field_size = params['field_size']
    embed_size = params['embed_size']
    feature_size = params['feature_size']
    deep_input_size = field_size * embed_size

    layers = params['layers']
    layer_num = len(layers)
    dropouts = params['dropouts']
    learning_rate = params['learning_rate']
    l2_reg = params['l2_reg']
    optimizer_type = params['optimizer_type']

    feat_val = tf.cast(features['value'], tf.float32)
    feat_idx = features['index']
    labels = tf.cast(labels, tf.float32)
    weights = build_weight(feature_size, embed_size)

    # ---------------------- FM Component ---------------------- #
    # First Order Term
    with tf.variable_scope('First-Order'):
        weight = tf.nn.embedding_lookup(weights['FM_W'], feat_idx)  # None * F
        y_first = tf.reduce_sum(tf.multiply(weight, feat_val),
                                1)  # None * F X None * F  ->  None * F (element-wise)  ->  None (sum)

    # Second Order Term
    with tf.variable_scope('Second-Order'):
        val = tf.reshape(feat_val, shape=[-1, field_size, 1])  # None * F  ->  None * F * 1
        embed = tf.nn.embedding_lookup(weights['FM_V'], feat_idx)  # None * F * K
        embed = tf.multiply(embed, val)  # None * F * K X None * F * 1  ->  None * F * K

        fm_sum_square = tf.square(tf.reduce_sum(embed, 1))  # None * F * K  ->  None * K
        fm_square_sum = tf.reduce_sum(tf.square(embed), 1)  # None * F * K  ->  None * K
        y_second = 0.5 * tf.reduce_sum(tf.subtract(fm_sum_square, fm_square_sum), 1)  # None * K -> None

    # --------------------- Deep Component --------------------- #
    with tf.variable_scope('Deep-Part'):
        deep_inputs = tf.reshape(embed, shape=[-1, deep_input_size])
        for i in range(layer_num):
            deep_inputs = tf.contrib.layers.fully_connected(
                inputs=deep_inputs,
                num_outputs=layers[i],
                activation_fn=tf.nn.relu,
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                scope='mlp%d' % i
            )
            deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropouts[i])

        deep_outputs = tf.contrib.layers.fully_connected(
            inputs=deep_inputs,
            num_outputs=1,
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
            scope='deep_out'
        )
        y_deep = tf.reshape(deep_outputs, shape=[-1])

    # ------------------------- DeepFM ------------------------- #
    with tf.variable_scope('DeepFM-Out'):
        y_bias = weights['FM_B'] * tf.ones_like(y_deep, dtype=tf.float32)
        y = y_bias + y_first + y_second + y_deep
        pred = tf.sigmoid(y, name='prediction')

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"prob": pred}
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)

    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels))
        loss += l2_reg * (tf.nn.l2_loss(weights['FM_W']) + tf.nn.l2_loss(weights['FM_V']))
        train_op = build_optimizer(optimizer_type, learning_rate, loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        predictions = {"prob": pred}
        eval_metric_ops = {"auc": tf.metrics.auc(labels, pred)}
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

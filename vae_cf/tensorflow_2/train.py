import tensorflow as tf


def compute_loss(x, model, anneal=tf.constant(1.0)):
    mean, logvar, KL = model.encode(x)
    z = model.reparameterize(mean, logvar)
    logits = model.decode(z)

    log_softmax_var = tf.nn.log_softmax(logits)
    neg_ll = -tf.reduce_mean(tf.reduce_sum(
        log_softmax_var * x, axis=-1))
    reg_loss = tf.math.add_n(model.losses)
    neg_ELBO = neg_ll + anneal * KL + reg_loss
    return neg_ELBO, neg_ll, KL


@tf.function
def train_step(inputs, model, optimizer, metrics, anneal=tf.constant(1.0)):
    with tf.GradientTape() as tape:
        neg_ELBO, neg_ll, KL = compute_loss(inputs, model, anneal)
    gradients = tape.gradient(neg_ELBO, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    metrics['neg_elbo'].update_state(neg_ELBO)
    metrics['neg_ll'].update_state(neg_ll)
    metrics['kl'].update_state(KL)


@tf.function
def test_step(inputs, model, metrics):
    features, labels = inputs
    zero = tf.constant(0, dtype=tf.float32)
    neg_val = tf.constant(-1000, dtype=tf.float32)

    logits = model.call(features)['logits']
    condition = tf.not_equal(features, zero)
    predictions = tf.where(condition, neg_val, logits)

    metrics['ndcg'].update_state(labels, predictions)


def train_epoch(
        train_datasets,
        eval_datasets,
        model,
        optimizer,
        metrics,
        num_epochs,
        anneal_cap,
        total_anneal_steps,
        summary_writer):
    step = 0
    for epoch in range(1, num_epochs + 1):
        for inputs in train_datasets:
            anneal = tf.constant(min(anneal_cap, 1. * step / total_anneal_steps))
            train_step(inputs, model, optimizer, metrics, anneal)
            step += 1

        for test_inputs in eval_datasets:
            test_step(test_inputs, model, metrics)

        with summary_writer.as_default():
            tf.summary.scalar('neg_ELBO_train_step', metrics['neg_elbo'].result(), step=epoch)
            tf.summary.scalar('neg_multi_ll_train_step', metrics['neg_ll'].result(), step=epoch)
            tf.summary.scalar('KL', metrics['kl'].result(), step=epoch)
            tf.summary.scalar('ndcg_at_k', metrics['ndcg'].result(), step=step)

        metrics['neg_elbo'].reset_states()
        metrics['neg_ll'].reset_states()
        metrics['kl'].reset_states()
        metrics['ndcg'].reset_states()

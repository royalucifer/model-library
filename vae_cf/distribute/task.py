import os
import argparse

import tensorflow as tf
import tensorflow_ranking as tfr

from .utils import Object
from .data_process import load_data
from .model import VAE
from .train import train_epoch
from .inputs import get_train_data, get_eval_data

strategy = tf.distribute.MirroredStrategy()


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--predict_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--tensorboard_dir', type=str, required=True)
    parser.add_argument('--job-dir', type=str, required=True)

    parser.add_argument('--anneal_cap', type=float, default=0.2)
    parser.add_argument('--total_anneal_steps', type=int, default=200000)
    parser.add_argument('--drop_prob', type=float, default=0.5)

    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--eval_batch_size', type=int, default=2000)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--random_seed', type=int, default=98765)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--l2_reg', type=float, default=0.0)
    return parser


def generate_params(args, n_items):
    model_params = {
        'p_dims': [200, 600, n_items],
        'q_dims': [n_items, 600, 200],
        'drop_prob': args.drop_prob,
        'l2_reg': args.l2_reg,
        'seed': args.random_seed}
    train_params = {
        'metrics': {
            'neg_elbo': tf.keras.metrics.Mean(name='train_neg_elbo'),
            'kl': tf.keras.metrics.Mean(name='train_kl'),
            'neg_ll': tf.keras.metrics.Mean(name='train_neg_ll'),
            'ndcg': tfr.keras.metrics.NDCGMetric(name='test_ndcg', topn=100)},
        'num_epochs': args.num_epochs,
        'anneal_cap': args.anneal_cap,
        'total_anneal_steps': args.total_anneal_steps,
        'summary_writer': tf.summary.create_file_writer(args.tensorboard_dir)}
    return model_params, train_params


def main():
    args = create_parser().parse_args()

    # Data
    tr_data, vd_data, map_dict = load_data(args.input_dir)
    _, n_items = tr_data.shape
    vd_features, vd_labels = vd_data
    tr_datasets = get_train_data(tr_data, 1, args.batch_size, strategy)
    vd_datasets = get_eval_data(vd_features, vd_labels, args.eval_batch_size, strategy)

    # Model
    with strategy.scope():
        model_params, train_params = generate_params(args, n_items)
        model = VAE(**model_params)
        optimizer = tf.keras.optimizers.Adam(args.learning_rate, epsilon=1e-08)
    train_epoch(tr_datasets, vd_datasets, model, optimizer, strategy, **train_params)

    # Saving
    tf.saved_model.save(
        obj=model,
        export_dir=os.path.join(args.predict_dir, 'model'),
        signatures=model.call.get_concrete_function(
            tf.TensorSpec(shape=[None, n_items], dtype=tf.float32, name="inputs")))
    Object.save(tr_data, os.path.join(args.model_dir, 'interactions.pkl'))
    Object.save(map_dict, os.path.join(args.model_dir, 'map.pkl'))


if __name__ == "__main__":
    main()

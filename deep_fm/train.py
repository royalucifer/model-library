import os
from multiprocessing import Pool

import tensorflow as tf

from .model import model_fn
from .data_process import _create_tfrecord
from .datasets import input_fn

MODEL_DIR = "AdModel/"
TRAIN_DIR = "TrainData/*.tfrecords"
TEMP_FILES = os.listdir("TrainTemp/")
PARAMS = {
    "feature_size": 838,
    "field_size": 128,
    "embed_size": 10,
    "layers": [400, 400, 400],
    "dropouts": [0.9, 0.9, 0.9],
    "l2_reg": 0.0001,
    "optimizer_type": "Adam",
    "learning_rate": 0.05}


if __name__ == "__main__":
    pool = Pool(25)
    data = pool.map(_create_tfrecord, TEMP_FILES)
    pool.close()
    pool.join()

    config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(log_device_placement=True),
        log_step_count_steps=100,
        save_summary_steps=10,
        keep_checkpoint_max=5)

    DeepFM = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=MODEL_DIR, params=PARAMS, config=config)
    DeepFM.train(
        input_fn=lambda: input_fn(TRAIN_DIR, num_epochs=5, batch_size=5000))

import os

import numpy as np
import pandas as pd
import pandavro as pdx
import tensorflow as tf

from .data_process import _create_training_data
from .model import model_fn
from .datasets import training_input_fn, seriving_input_fn

MODEL_DIR = "RecModel/"
TRAIN_DIR = "TrainData/*.tfrecords.gz"
TEMP_FILES = os.listdir("TrainTemp/")

NUM_EPOCHS = 1
BATCH_SIZE = 1000000
PARAMS = {
    "user_feats_size": 8222243,
    "item_feats_size": 343419,
    "embed_size": 150,
    "optimizer_type": "Adam",
    "learning_rate": 0.05,
    "l2_reg": 0.0001}


if __name__ == "__main__":
    # load data
    data = pd.concat([pdx.read_avro("TrainTemp/" + f) for f in TEMP_FILES])
    data = np.hstack((
        data["user_index"].values[:, None],
        data["item_indexs"].values[:, None]))
    _create_training_data(data)

    config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(
            device_count={"GPU": 1},
            log_device_placement=True),
        log_step_count_steps=100,
        save_summary_steps=10,
        keep_checkpoint_max=3)

    BPR = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=MODEL_DIR, params=PARAMS, config=config)
    BPR.train(input_fn=lambda: training_input_fn(TRAIN_DIR, num_epochs=1, batch_size=1000000))
    prediction = BPR.predict(input_fn=lambda: seriving_input_fn("item.csv", user_id=6649401))

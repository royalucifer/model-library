import pickle

import pandas as pd
import pandavro as pdx
from apache_beam.io.gcp import gcsio


class Object:
    @staticmethod
    def _save_avro(obj, dirs):
        gcs = gcsio.GcsIO()
        with gcs.open(dirs, 'wb') as f:
            pdx.to_avro(f, obj)

    @staticmethod
    def _save_pickle(obj, dirs):
        gcs = gcsio.GcsIO()
        with gcs.open(dirs, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def save(cls, obj, dirs, data_type='pickle'):
        if data_type == 'avro':
            return cls._save_avro(obj, dirs)
        elif data_type == 'pickle':
            return cls._save_pickle(obj, dirs)

    @staticmethod
    def _load_avro(dirs):
        gcs = gcsio.GcsIO()
        with gcs.open(dirs, 'rb') as f:
            return pdx.from_avro(f)

    @staticmethod
    def _load_pickle(dirs):
        gcs = gcsio.GcsIO()
        with gcs.open(dirs, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def _load_pandas_pickle(dirs):
        gcs = gcsio.GcsIO()
        with gcs.open(dirs, 'rb') as f:
            return pd.read_pickle(f)

    @classmethod
    def load(cls, dirs, data_type='pickle'):
        if data_type == 'avro':
            return cls._load_avro(dirs)
        elif data_type == 'pandas_pickle':
            return cls._load_pandas_pickle(dirs)
        elif data_type == 'pickle':
            return cls._load_pickle(dirs)

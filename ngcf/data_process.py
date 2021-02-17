import numpy as np
import pandas as pd
import scipy as sp

from pickle_obj import Obj

PATH = '/home/jupyter/RecData/0728/'


def load_data(file_path):
    data = pd.read_pickle(file_path)
    id_map = dict()
    for i in ['clientid', 'pid']:
        cat = data[i].astype('category').cat
        data[i] = cat.codes
        id_map[i] = cat.categories
    return data, id_map


def to_csr_matrix(data, n_users, n_items):
    interactions = sp.csr_matrix(
        (data['score'], (data['clientid'], data['pid'])),
        dtype=np.double,
        shape=(n_users, n_items))
    return interactions


def to_adj_martix(data, n_users, n_items):
    def normalized(adj_mat):
        rowsum = np.array(adj_mat.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj_mat)
        return norm_adj.tocsr()

    row = pd.concat([tr_data['clientid'], tr_data['pid'] + n_users], ignore_index=True)
    col = pd.concat([tr_data['pid'] + n_users, tr_data['clientid']], ignore_index=True)
    val = [1] * row.shape[0]

    adj_mat = sp.coo_matrix(
        (val, (row, col)),
        dtype=np.double,
        shape=(n_users + n_items, n_users + n_items))

    norm_adj_mat = normalized(adj_mat) + sp.eye(adj_mat.shape[0])
    return norm_adj_mat


class DataGenerator:
    def __init__(self, n_users, n_items, n_heldout):
        self.n_users = n_users
        self.n_items = n_items
        self.n_heldout = n_heldout

    def _get_heldout_uid(self):
        np.random.seed(98765)
        random_uid = np.random.permutation(self.n_users)
        heldout_uid = random_uid[:self.n_heldout]
        heldout_uid.sort()
        return heldout_uid

    def _convert_user_index(self, df_series):
        cat = df_series.astype('category').cat
        return cat.codes

    def _split_valid_dataset(self, df, test_prob=0.2):
        tr_list, te_list = [], []
        for _, group in df.groupby('clientid'):
            clicked_items = len(group)
            n_test_items = int(test_prob * clicked_items)

            idx = np.zeros(clicked_items, dtype='bool')
            idx[np.random.choice(clicked_items, size=n_test_items, replace=False)] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])

        vd_data_tr = pd.concat(tr_list)
        vd_data_te = pd.concat(te_list)
        return vd_data_tr, vd_data_te

    def split_train_test(self, data, test_prob=0.2):
        test_uid = self._get_heldout_uid()
        test_df = data.loc[data['clientid'].isin(test_uid)]
        train_df = data.loc[~data['clientid'].isin(test_uid)]

        vd_data_tr, vd_data_te = self._split_valid_dataset(test_df, test_prob)
        tr_data = pd.concat([train_df, vd_data_tr])

        vd_data_tr['clientid'] = self._convert_user_index(vd_data_tr['clientid'])
        vd_data_te['clientid'] = self._convert_user_index(vd_data_te['clientid'])
        return tr_data, vd_data_tr, vd_data_te, test_uid


if __name__ == "__main__":
    data, id_map = load_data('/home/jupyter/RecData/RecData.pkl')
    n_users = len(id_map['clientid'])
    n_items = len(id_map['pid'])
    n_heldout = 20000

    data_generator = DataGenerator(n_users, n_items, n_heldout)
    tr_data, vd_data_tr, vd_data_te, vd_uid = data_generator.split_train_test(data)

    norm_adj_mat = to_adj_martix(tr_data, n_users, n_items)
    tr_data_csr = to_csr_matrix(tr_data, n_users, n_items)
    vd_data_tr_csr = to_csr_matrix(vd_data_tr, n_heldout, n_items)
    vd_data_te_csr = to_csr_matrix(vd_data_te, n_heldout, n_items)

    user_dict = dict(zip(id_map['clientid'], range(n_users)))
    item_dict = dict(zip(range(n_items), id_map['pid']))

    # NGCF
    tr_data.to_pickle(PATH + 'NGCF/pair_tr_data.pkl')
    sp.sparse.save_npz(PATH + 'NGCF/norm_adj_mat.npz', norm_adj_mat)
    sp.sparse.save_npz(PATH + 'NGCF/sparse_vd_data_tr.npz', vd_data_tr_csr)
    sp.sparse.save_npz(PATH + 'NGCF/sparse_vd_data_te.npz', vd_data_te_csr)
    Obj.save(vd_uid, PATH + 'NGCF/vd_uid.pkl')
    Obj.save(user_dict, PATH + 'NGCF/user_dict.pkl')
    Obj.save(item_dict, PATH + 'NGCF/item_dict.pkl')

    # VAE
    sp.sparse.save_npz(PATH + 'VAE/sparse_tr_data.npz', tr_data_csr)
    sp.sparse.save_npz(PATH + 'VAE/sparse_vd_data_tr.npz', vd_data_tr_csr)
    sp.sparse.save_npz(PATH + 'VAE/sparse_vd_data_te.npz', vd_data_te_csr)
    Obj.save(user_dict, PATH + 'VAE/user_dict.pkl')
    Obj.save(item_dict, PATH + 'VAE/item_dict.pkl')

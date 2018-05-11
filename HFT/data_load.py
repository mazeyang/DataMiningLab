import numpy as np
from scipy.sparse import csr_matrix
import pickle


def review_data(review_data_filename=r'Data\reviews.npz'):
    loader = np.load(review_data_filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def ratings_data(ratings_data_filename=r'Data\ratings.npz'):
    loader = np.load(ratings_data_filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def user_ids(user_id_filename=r'Data\user_ids.pkl'):
    return pickle.load(open(user_id_filename, 'rb'))


def business_ids(business_id_filename=r'Data\business_ids.pkl'):
    return pickle.load(open(business_id_filename, 'rb'))


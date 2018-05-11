import numpy as np
import data_load


class RatingModel:

    def __init__(self, ratings_filename=r'Data\ratings.npz', n_hidden_factors=10):
        self.data = data_load.ratings_data(ratings_filename)
        self.n_users, self.n_items = self.data.shape
        self.n_hidden_factors = n_hidden_factors
        self.corpus_ix = self.data.nonzero()

        # parameters
        self.alpha = np.random.uniform()
        self.beta_user = np.random.rand(self.n_users)
        self.beta_item = np.random.rand(self.n_items)
        self.gamma_user = np.random.rand(self.n_users, self.n_hidden_factors)
        self.gamma_item = np.random.rand(self.n_items, self.n_hidden_factors)
        self.predicted_rating = np.zeros((self.n_users, self.n_items))

    def get_predicted_ratings(self):
        # Latent-Factor Recommender Systems
        # calculate rec(u, i) - formula 1
        for u, i in zip(self.corpus_ix[0], self.corpus_ix[1]):
            self.predicted_rating[u, i] = np.dot(self.gamma_user[u, :], self.gamma_item[i, :]) + self.alpha + \
                                          self.beta_user[u] + self.beta_item[i]
        # self.predicted_rating += self.alpha + self.beta_user[:, None] + self.beta_item
        # self.predicted_rating[np.logical_not(self.corpus_ix)] = 0

    def get_rating_error(self):
        # calculate the error
        return np.sum(np.square(self.predicted_rating - self.data))


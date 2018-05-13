import numpy as np
from datetime import datetime as dt
from scipy.optimize import minimize

from ratingModel import RatingModel
from reviewModel import ReviewModel
import data_pre_process
import data_load


class HFT:
    def __init__(self, ratings_filename=r'Data\ratings.npz', reviews_filename=r'Data\reviews.npz', n_hidden=10):
        self.rating_model = RatingModel(ratings_filename, n_hidden)
        self.rating_model.get_predicted_ratings()

        self.kappa = np.random.uniform()

        self.review_model = ReviewModel(reviews_filename, n_hidden)
        self.get_theta()
        self.review_model.Gibbsampler()

        self.mu = 0.01

        self.opt_iter = 0
        self.max_iter = 10
        self.step_size = 1e-5
        self.max_kappa = 1e+5
        self.max_grad_iter = 100
        self.convergence_threshold = 0.01  # to be changed

        self.user_dict = data_load.user_ids()
        self.business_dict = data_load.business_ids()

    def get_theta(self):
        # get theta(i, k) -- formula 4
        self.review_model.theta = np.exp(self.kappa * self.rating_model.gamma_item)
        partition = np.sum(self.review_model.theta, axis=1)
        self.review_model.theta /= partition[:, None]

    def flatten(self):
        return np.concatenate((np.array([self.rating_model.alpha]),
                               self.rating_model.beta_user,
                               self.rating_model.beta_item,
                               self.rating_model.gamma_user.flatten(),
                               self.rating_model.gamma_item.flatten(),
                               self.review_model.phi.flatten(),
                               np.array([self.kappa])))

    def structure(self, array):
        param_list = list()
        offset = 0

        # alpha
        param_list += [array[0]]
        offset += 1

        # beta_user
        ptr = offset + self.rating_model.n_users
        param_list += [array[offset:ptr]]
        offset = ptr

        # beta_item
        ptr = offset + self.rating_model.n_items
        param_list += [array[offset:ptr]]
        offset = ptr

        # gamma_user
        ptr = offset + self.rating_model.n_users*self.rating_model.n_hidden_factors
        param_list += [array[offset:ptr].reshape(self.rating_model.n_users, self.rating_model.n_hidden_factors)]
        offset = ptr

        # gamma_item
        ptr = offset + self.rating_model.n_items*self.rating_model.n_hidden_factors
        param_list += [array[offset:ptr].reshape(self.rating_model.n_items, self.rating_model.n_hidden_factors)]
        offset = ptr

        # phi
        ptr = offset + self.review_model.n_topics * self.review_model.n_vocab
        param_list += [array[offset:ptr].reshape(self.review_model.n_topics, self.review_model.n_vocab)]
        offset = ptr

        # kappa
        param_list += [array[offset]]

        return tuple(param_list)

    def get_gradients(self):
        rating_loss = self.rating_model.predicted_rating - self.rating_model.data
        review_loss = (self.review_model.topic_frequencies -
                       self.review_model.theta * np.sum(self.review_model.topic_frequencies, axis=1)[:, None])

        alpha_gradient = 2 * np.sum(rating_loss)
        beta_item_gradients = 2 * np.ravel(np.sum(rating_loss, axis=0))
        beta_user_gradients = 2 * np.ravel(np.sum(rating_loss, axis=1))
        gamma_user_gradients = 2 * np.dot(rating_loss, self.rating_model.gamma_item)
        gamma_item_gradients = 2 * np.dot(rating_loss.transpose(), self.rating_model.gamma_user) - \
                               self.mu * self.kappa * review_loss

        phi = np.exp(self.review_model.phi + self.review_model.backgroundwords[None, :])
        phi /= phi.sum(axis=1)[:, None]
        topic_counts = self.review_model.topic_frequencies.sum(axis=0)
        phi_gradients = - self.mu * (self.review_model.word_topic_frequencies - topic_counts[None, :].transpose() *
                                     phi)
        # phi_gradients = np.divide(self.review_model.word_topic_frequencies, self.review_model.phi)
        kappa_gradient = np.sum(self.rating_model.gamma_item * review_loss)

        return [alpha_gradient, beta_user_gradients, beta_item_gradients,
                gamma_user_gradients, gamma_item_gradients, phi_gradients, kappa_gradient]

    def get_error(self):
        # the feature: calculate f -- formula 5
        return self.rating_model.get_rating_error() - self.mu * self.review_model.loglikelihood()

    def gradient_update(self):
        self.opt_iter += 1
        gradients = self.get_gradients()

        self.rating_model.alpha -= self.step_size * gradients[0]
        self.rating_model.beta_user -= self.step_size * gradients[1]
        self.rating_model.beta_item -= self.step_size * gradients[2]
        self.rating_model.gamma_user -= self.step_size * gradients[3]
        self.rating_model.gamma_item -= self.step_size * gradients[4]
        self.review_model.phi -= self.step_size * gradients[5]
        self.kappa -= self.step_size * gradients[6]

        self.get_theta()
        self.rating_model.get_predicted_ratings()
        return True

    def train(self):
        num_iter = 0
        while num_iter < self.max_grad_iter:
            previous_params = HFT.flatten(self)
            # print(hft.flatten())
            status = self.gradient_update()
            if not status:
                break
            if np.sum(np.abs(previous_params - HFT.flatten(self))) <= self.convergence_threshold:
                break
            num_iter += 1

    def learn(self):
        num_iter = 0

        while num_iter < self.max_iter:
            num_iter += 1

            previous_params = HFT.flatten(self)

            self.train()
            # exp and normalize phi
            self.review_model.phi += self.review_model.backgroundwords
            self.review_model.phi = np.exp(self.review_model.phi)
            self.review_model.phi /= np.sum(self.review_model.phi, axis=1)[:, None]

            if np.sum(np.abs(previous_params - HFT.flatten(self))) <= self.convergence_threshold:
                break

            self.review_model.Gibbsampler()
            print(num_iter, self.get_error())
            # print 'Main iteration', num_iter
        self.rating_model.get_predicted_ratings()

    def predict(self, u, i):
        res = np.dot(self.rating_model.gamma_user[u, :], self.rating_model.gamma_item[i, :]) + \
                                                   self.rating_model.alpha + \
                                                   self.rating_model.beta_user[u] + self.rating_model.beta_item[i]
        # self.rating_model.get_predicted_ratings()
        return res

    def error_gradients(self, params):
        self.opt_iter += 1
        start_time = dt.now()

        alpha, beta_user, beta_item, gamma_user, gamma_item, phi, kappa = self.structure(params)

        # get rating error
        predicted_rating = np.zeros((self.rating_model.n_users, self.rating_model.n_items))
        for u, i in zip(self.rating_model.corpus_ix[0], self.rating_model.corpus_ix[1]):
            predicted_rating[u, i] = np.dot(gamma_user[u, :], gamma_item[i, :]) + alpha + beta_user[u] + beta_item[i]

        rating_loss = predicted_rating - self.rating_model.data
        rating_error = np.sum(np.square(rating_loss))
        rating_error_time = (dt.now() - start_time).seconds

        # get log_likelihood
        theta = np.exp(kappa * gamma_item)
        theta /= np.sum(theta, axis=1)[:, None]

        log_likelihood = 0
        for i in range(self.review_model.n_docs):
            words = self.review_model.reviews[i]
            topics = self.review_model.z[i]
            log_likelihood += np.sum(np.log(theta[i, topics]) + np.log(phi[topics, words]))
        review_error_time = (dt.now() - start_time).seconds - rating_error_time

        error = rating_error - self.mu * log_likelihood

        # get gradients
        review_loss = (self.review_model.topic_frequencies -
                       theta * np.sum(self.review_model.topic_frequencies, axis=1)[:, None])

        alpha_gradient = 2 * np.sum(rating_loss)
        beta_item_gradients = 2 * np.ravel(np.sum(rating_loss, axis=0))
        beta_user_gradients = 2 * np.ravel(np.sum(rating_loss, axis=1))
        gamma_user_gradients = 2 * np.asarray(np.dot(rating_loss, gamma_item))
        gamma_item_gradients = 2 * np.asarray(np.dot(rating_loss.transpose(), gamma_user)) - \
                               self.mu * kappa * review_loss
        phi_gradients = np.divide(self.review_model.word_topic_frequencies, phi)
        kappa_gradient = np.sum(gamma_item * review_loss)

        # try:
        gradients = np.concatenate((np.array([alpha_gradient]), beta_user_gradients, beta_item_gradients,
                                   gamma_user_gradients.flatten(), gamma_item_gradients.flatten(),
                                   phi_gradients.flatten(), np.array([kappa_gradient])))

        print(self.opt_iter, rating_error_time, review_error_time, (dt.now() - start_time).seconds, start_time.time())
        return error, gradients

    def update(self):
        params = self.flatten()

        self.opt_iter = 0
        opt_params = minimize(self.error_gradients, params, method='L-BFGS-B', jac=True, options={'maxiter': 2, 'maxfun': 2, 'disp': True}).x
        self.rating_model.alpha, self.rating_model.beta_user, self.rating_model.beta_item, \
            self.rating_model.gamma_user, self.rating_model.gamma_item, \
            self.review_model.phi, self.kappa = self.structure(opt_params)
        self.review_model.phi += self.review_model.backgroundwords
        self.review_model.phi = np.exp(self.review_model.phi)
        self.review_model.phi /= self.review_model.phi.sum(axis=1)[:, None]


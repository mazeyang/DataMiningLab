import numpy as np
import random
import sys
import data_load


def flatten_bow(bow_review):
    """
    Flattens the bag of words array so that it is the
    same shape as the topic assingment array (z)
    Parameters
    ----------
    bow_review
    Returns
    -------
    words
    """
    indices = np.flatnonzero(bow_review)
    words = []
    for j in indices:
        words += [j] * int(bow_review[j])
    return np.array(words)


def sampleWithDistribution(p):
    """ Sampler that samples with respect to distribution p
    Parameters
    ----------
    p : numpy array
    Returns
    -------
    i: index of sampled value
    """
    r = random.random()  # Rational number between 0 and 1

    for i in range(len(p)):
        r = r - p[i]
        # print('p[i]: ', type(p[i]))
        if r <= 0:
            return i
    raise Exception("Error... selectWithDistribution with r value %f" % r)


class ReviewModel:
    def __init__(self, reviews_filename=r'Data\reviews.npz', n_topics=10):
        data = data_load.review_data(reviews_filename)
        self.n_docs, self.n_vocab = data.shape
        self.n_topics = n_topics

        self.phi = np.random.rand(n_topics, self.n_vocab)
        self.phi /= self.phi.sum(axis=1)[:, None]

        self.theta = np.random.rand(self.n_docs, n_topics)
        self.theta /= self.theta.sum(axis=1)[:, None]

        self.topic_frequencies = np.zeros((self.n_docs, self.n_topics))
        self.word_topic_frequencies = np.zeros((self.n_topics, self.n_vocab))
        self.backgroundwords = np.zeros(self.n_vocab)

        self.z = list()
        self.reviews = list()
        for doc_ix in range(self.n_docs):
            data_review = flatten_bow(data[doc_ix, :].toarray()[0])
            n_words = len(data_review)
            self.z.append(np.zeros(n_words, dtype=int))
            self.reviews.append(data_review)
            np.add.at(self.backgroundwords, data_review, 1.0)
        self.backgroundwords /= np.sum(self.backgroundwords)

    def loglikelihood(self):
        """Computes likelihood of a corpus
        Returns
        -------
        loglikelihood: The loglikelihood of the entire corpus
        """
        # All_loglikelihoods = list()
        log_likelihood = 0

        for i in range(self.n_docs):
            words = self.reviews[i]
            topics = self.z[i]

            # loglikelihood = np.log(self.theta[([i]*len(topics), topics)]) + np.log(self.phi[(topics, words)])
            log_likelihood += np.sum(np.log(self.theta[i, topics]) + np.log(self.phi[topics, words]))
            # if np.isnan(log_likelihood):
            #     print np.sum(np.log(self.theta[i, topics]) + np.log(self.phi[topics, words]))
            #     print i
            #     sys.exit(1)

            # All_loglikelihoods.append(np.sum(loglikelihood))

        # if log_likelihood - sum(All_loglikelihoods) != 0.0:
        #     print log_likelihood, sum(All_loglikelihoods)
        #     sys.exit(1)
        # return sum(All_loglikelihoods)
        return log_likelihood

    def Gibbsampler(self):
        """
        Resamples the topic_assingments accross the entires corpus
        Returns:
        new_topic_assingments: list of numpy arrays
        """

        new_topic_assignments = list()

        self.topic_frequencies.fill(0)
        self.word_topic_frequencies.fill(0)

        for i in range(self.n_docs):
            words = self.reviews[i]

            p = self.theta[i, :] * self.phi[:, words].transpose()
            p /= np.sum(p, axis=1)[:, None]
            p = p.tolist()
            # print('p: ', type(p))
            # print(p)
            # topic_assignments = sampleWithDistribution(p)
            topic_assignments = map(sampleWithDistribution, p)
            # print(topic_assignments)
            topic_assignments = list(topic_assignments)
            # print('1: ', type(topic_assignments))
            # print('2: ', type(words))
            np.add.at(self.topic_frequencies[i], topic_assignments, 1)
            np.add.at(self.word_topic_frequencies, [topic_assignments, words], 1)

            new_topic_assignments.append(np.array(topic_assignments))

        self.z = new_topic_assignments


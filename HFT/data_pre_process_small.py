import os
from pprint import pprint as pp
import nltk
from scipy.sparse import lil_matrix, csr_matrix
import numpy as np
import pandas as pd
import pickle
from datetime import datetime as dt


class DataPreProcess:
    def __init__(self):
        self.input_file = r'Data\train.dat'
        self.output_file = r'Data\reviews.csv'
        self.test_input_file = r'Data\test.dat'
        self.test_output_file = r'Data\to_predict.csv'
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.stemmer = nltk.stem.snowball.SnowballStemmer('english')
        self.lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
        self.cnt = 0  # total 393366
        self.words_cnt = 0

    def load_data(self, input_file=r'Data\train.dat', output_file=r'Data\reviews.csv',
                  test_input_file=r'Data\test.dat', test_output_file=r'Data\to_predict.csv'):
        self.input_file = input_file
        self.output_file = output_file
        self.test_input_file = test_input_file
        self.test_output_file = test_output_file
        record = [0 for x in range(6)]
        print(record)
        # load train file
        file = open(self.input_file, 'rU', encoding='UTF-8')
        data = []
        # item_ids = []
        for line in file:
            self.cnt += 1
            if self.cnt >= 100000:
                print('solved', self.cnt, ' cur time:', dt.now())
                break
            line = line.split(' ')
            user = line[0]
            item = line[1]
            rating = line[2]
            record[int(rating)] += 1
            word_num = line[3]
            if word_num == 0 or len(line) < 5:
                print('hhhhhhhhhhhhhhhhhhhh')
                self.cnt -= 1
                continue
            review = ' '.join(line[4:])
            # item_ids.append(item)
            data.append([item, rating, self._process_text(review), user])
        file.close()
        print('cnt: ', self.cnt)
        print(record)
        data = pd.DataFrame(data)
        data.columns = ['business_id', 'stars', 'text', 'user_id']
        data.to_csv(output_file, encoding='utf-8', line_terminator='\n')

        # load test file
        # file = open(self.test_input_file, 'rU', encoding='UTF-8')
        # data = []
        # for line in file:
        #     line = line.split(' ')
        #     user = line[0]
        #     item = line[1]
        #     data.append([user, item])
        # file.close()
        # data = pd.DataFrame(data)
        # data.columns = ['user_id', 'business_id']
        # data.to_csv(test_output_file, encoding='utf-8')

    def _process_text(self, sentence):
        '''
        pre_precess the sentence(1. remove stopwords and 2. stem)
        :param sentence:
        :return:
        '''
        sentence = nltk.word_tokenize(sentence.lower())  # Tokenize
        words = [w for w in sentence if w.isalpha() and w not in self.stopwords]  # Remove stopwords
        stemmed_words = [self.stemmer.stem(self.lmtzr.lemmatize(w)) for w in words]  # Stem and Lemmatize
        # res_words = [w for w in stemmed_words if w.isalpha() and w not in stopwords]
        return ' '.join(stemmed_words)

    def vocab_build(self):
        reviews = pd.read_csv(self.output_file)
        review_text = reviews['text']
        # Building vocabulary...
        self._build_vocab(review_text)

        # Loading dictionary...
        vocab_lookup = {}
        with open(r'Data\vocab.txt', 'r') as f:
            for line in f.readlines():
                lookup_int, key, __ = line.split(',')
                vocab_lookup[key] = lookup_int
        # Building bag of words...
        reviews['text'] = review_text.apply(lambda row: self._bag_of_word(row, vocab_lookup))
        reviews = reviews[reviews['text'] != '']
        reviews.to_csv(r'Data\bow_reviews.csv')

    def _build_vocab(self, text):
        """Takes a dataframe of texts and creates a vocab file
        vocab.txt"""
        # print('begin: \n', text, 'end.\n')
        # print(type(text))
        vocab = {}
        for indexs in text.index:

            entry = text.loc[indexs]
            if not isinstance(entry, str):
                # self.cnt -= 1
                continue
            # print(indexs, type(entry), entry)
            for word in entry.split():
                try:
                    vocab[word] += 1
                except:
                    vocab[word] = 1
    
        filtered_vocab = {}
        for key in vocab.keys():
            if vocab[key] >= 500:
                filtered_vocab[key] = vocab[key]

        self.words_cnt = len(filtered_vocab)

        with open(r'Data\vocab.txt', 'w', encoding='utf-8') as f:
            i = 0
            for key in sorted(filtered_vocab.keys()):
                f.write("{0},{1},{2}\n".format(i, key, filtered_vocab[key]))
                i += 1
        # print(vocab)

    def _bag_of_word(self, text, vocab_lookup):
        counts = {}

        try:
            for word in text.split():
                if word in counts:
                    counts[word] += 1
                elif word in vocab_lookup:
                    counts[word] = 1
        except:
            print(text)

        return ' '.join(str(vocab_lookup[word]) + ":" + str(counts[word]) for word in counts.keys())

    def matrix_make(self):
        self.review_data = pd.read_csv(r'Data\bow_reviews.csv')
        self.n_reviews = self.cnt
        self.n_words = self.words_cnt
        print('reviews cnt:', self.n_reviews)
        print('words cnt:', self.n_words)

        # Mapping all business_ids
        self.business_ids = list()
        for business in self.review_data['business_id']:
            self.business_ids.append(business)
        unique_business_ids = list(set(self.business_ids))
        self.n_businesses = len(unique_business_ids)

        self.business_dict = dict()
        for index, b_id in enumerate(unique_business_ids):
            self.business_dict[index] = b_id
            self.business_dict[b_id] = index

        # Mapping all user_ids
        self.user_ids = list()
        for user in self.review_data['user_id']:
            self.user_ids.append(user)
        unique_user_ids = list(set(self.user_ids))
        self.n_users = len(unique_user_ids)

        self.user_dict = dict()
        for index, u_id in enumerate(unique_user_ids):
            self.user_dict[index] = u_id
            self.user_dict[u_id] = index

        self.reviews = lil_matrix((self.n_businesses, self.n_words))
        self.ratings = lil_matrix((self.n_users, self.n_businesses))

        # operate
        print('1..')
        self._get_reviews()
        print('2..')
        self._save_reviews()
        print('3..')
        self._get_ratings()
        print('4..')
        self._save_ratings()
        print('5..')
        self._save_ids()

    def _get_reviews(self):
        i = 0
        for r_ix, review in enumerate(self.review_data['text']):
            b_id = self.business_ids[r_ix]
            b_ix = self.business_dict[b_id]
            # if i % 10000 == 0:
            #     print('solved :', i)
            i += 1
            words = review.split()
            for word in words:
                word = word.split(':')
                self.reviews[b_ix, int(word[0])] += int(word[1])

    def _get_ratings(self):
        rating_counts = lil_matrix(self.ratings.shape)
        for r_ix, rating in enumerate(self.review_data['stars']):
            b_id = self.business_ids[r_ix]
            b_ix = self.business_dict[b_id]

            u_id = self.user_ids[r_ix]
            u_ix = self.user_dict[u_id]

            # maybe a user review an item for several times
            if rating_counts[u_ix, b_ix] == 0:
                rating_counts[u_ix, b_ix] = 1
                self.ratings[u_ix, b_ix] = rating
            else:
                prior_rating = self.ratings[u_ix, b_ix]
                prior_count = rating_counts[u_ix, b_ix]
                new_count = prior_count+1
                new_rating = (prior_rating*prior_count + rating) / new_count

                rating_counts[u_ix, b_ix] = new_count
                self.ratings[u_ix, b_ix] = new_rating

    def _save_reviews(self, review_output_filename=r'Data\reviews.npz'):
        self.reviews = csr_matrix(self.reviews)
        np.savez(open(review_output_filename, 'wb'),
                 data=self.reviews.data, indices=self.reviews.indices, indptr=self.reviews.indptr,
                 shape=self.reviews.shape)

    def _save_ratings(self, ratings_output_filename=r'Data\ratings.npz'):
        self.ratings = csr_matrix(self.ratings)
        np.savez(open(ratings_output_filename, 'wb'),
                 data=self.ratings.data, indices=self.ratings.indices, indptr=self.ratings.indptr,
                 shape=self.ratings.shape)

    def _save_ids(self, business_output_filename=r'Data\business_ids.pkl', user_output_filename=r'Data\user_ids.pkl'):
        pickle.dump(self.business_dict, open(business_output_filename, 'wb'))
        pickle.dump(self.user_dict, open(user_output_filename, 'wb'))

    def run(self):
        start_time = dt.now()

        print('start load data...')
        self.load_data()
        print('data load success. cost time:', (dt.now() - start_time).seconds, 'seconds.')
        print('cnt:', self.cnt)

        print('start build vocab...')
        t1 = dt.now()
        self.vocab_build()
        print('vocab build success. cost time:', (dt.now() - t1).seconds, 'seconds.')

        print('start make matrix...')
        t2 = dt.now()
        self.matrix_make()
        print('data convert success. cost time:', (dt.now() - t2).seconds, 'seconds.')

        print('total cost time:', (dt.now() - start_time).seconds, 'seconds.')


if __name__ == '__main__':
    # load data and pre_process
    DataPreProcess().run()


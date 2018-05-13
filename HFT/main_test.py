from datetime import datetime as dt
import pandas as pd
from hft import HFT
from pprint import pprint
import random


if __name__ == '__main__':

    hft = HFT(ratings_filename=r'Data\ratings.npz', reviews_filename=r'Data\reviews.npz')

    start_time = dt.now()
    grads = hft.get_gradients()
    print('1. Finished get gradients in', (dt.now() - start_time).seconds, 'seconds')

    start_time = dt.now()
    hft.rating_model.get_predicted_ratings()
    print('2. Finished get predicted ratings in', (dt.now() - start_time).seconds, 'seconds')

    start_time = dt.now()
    hft.review_model.Gibbsampler()
    print('3. Finished performing Gibbs sampling in', (dt.now() - start_time).seconds, 'seconds')

    l = hft.review_model.loglikelihood()

    start_time = dt.now()
    hft.update()
    print('4. Finished updating parameters in', (dt.now() - start_time).seconds, 'seconds')

    start_time = dt.now()

    hft.learn()

    print('5. Finished updating parameters in', (dt.now() - start_time).seconds, 'seconds')

    # print(len(hft.business_dict[0]))

    file = open(r'Data\test.dat', 'rU', encoding='UTF-8')
    wrt = open(r'Data\result.dat', 'w', encoding='UTF-8')
    record = [0 for x in range(6)]
    print(record)
    for line in file:
        line = line.split(' ')
        u = line[0]
        i = line[1][:-1]  # remove '\n'

        if u in hft.user_dict.keys() and i in hft.business_dict.keys():
            unum = hft.user_dict[u]
            inum = hft.business_dict[i]
            rat = hft.predict(unum, inum)
            rat = int(rat + 0.5) if 0 <= rat <= 5 else (0 if rat < 0 else 5)
            record[rat] += 1
            s = ('%s %s %s\n' % (u, i, rat))
            wrt.write(s)

    file.close()
    wrt.close()
    print(record)


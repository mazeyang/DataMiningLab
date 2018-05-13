from datetime import datetime as dt
import pandas as pd
from hft import HFT
from pprint import pprint


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

    # for i in range(100):
    #
    #     previous_params = hft.flatten()
    #     status = hft.gradient_update()
    #     if not status:
    #         break
    #     diff = np.abs(previous_params - hft.flatten())
    #     print np.mean(diff)

        # break_flag = False
        # for j in range(10):
        #     status = hft.gradient_update()
        #     if not status:
        #         break_flag = True
        #         break
        # if break_flag:
        #     break
        # hft.review_model.Gibbsampler()
        #
        # difference = np.absolute(previous_params - hft.flatten())

        # print difference.max()
        # print difference

        # print i, ': Gibbs Sampling', hft.kappa
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
        # print(u, i)
        # if u in hft.user_dict.keys():
        #     print(u, 'haha', hft.user_dict[u])
        # if i in hft.business_dict.keys():
        #     print(u, 'ffff', hft.business_dict[u])
        # print(i)
        # print(len(i))
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
    # print(hft.business_dict[0])
    # print(len(hft.business_dict[0]))

    # pprint('\n------------------------------')
    # mat = hft.rating_model.data.toarray()
    # pprint(mat[878:880, 42:44])
    # pprint('------------------------------')

    # test_user = 'R6vb0FtmClhfwajs_AuusQ'
    # test_item = 'jQsNFOzDpxPmOurSWCg1vQ'
    # test_user = hft.user_dict[test_user]
    # test_item = hft.business_dict[test_item]
    # for u in hft.user_dict.keys():
    #     for i in hft.business_dict.keys():
    #         if str(u).isdigit() and str(i).isdigit() and mat[u-1][i-1] == 0:
    #             test_user, test_item = u, i
    #             rating = mat[u-1][i-1]
    # print('test: ', test_user, test_item, rating)
    # x = hft.predict(test_user, test_item)

    # m = hft.rating_model.predicted_rating
    # a, b = m.shape
    # print('shape', a, b)
    #
    # c = 0
    # for i in range(a):
    #     for j in range(b):
    #         if m[i][j] != 0:
    #             c += 1
    #             # if c > 10:
    #             #     break
    #             print(i, j, m[i][j])
    # print(c)

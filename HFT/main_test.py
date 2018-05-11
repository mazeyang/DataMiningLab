from datetime import datetime as dt

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

    pprint('\n------------------------------')
    pprint(hft.user_dict)
    pprint('------------------------------')

    test_user = 'R6vb0FtmClhfwajs_AuusQ'
    test_item = 'jQsNFOzDpxPmOurSWCg1vQ'
    test_user = hft.user_dict[test_user]
    test_item = hft.business_dict[test_item]
    r = hft.predict(test_user, test_item)
    print(r)


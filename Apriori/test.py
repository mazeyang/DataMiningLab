from Apriori import Apriori
from optparse import OptionParser
import time


def loadData(filepath):
    file = open(filepath, 'rU')
    data = []
    for line in file:
        tmp = line.split(' ')
        data.append(tmp)
    file.close()
    return data


if __name__ == '__main__':

    optparser = OptionParser()
    optparser.add_option('-s', '--minSupport',
                         dest='minS',
                         help='minimum support value. e.g. 0.8',
                         default=0.6,
                         type='float')
    optparser.add_option('-c', '--minConfidence',
                         dest='minC',
                         help='minimum confidence value. e.g. 0.8',
                         default=0.8,
                         type='float')
    optparser.add_option('-i', '--inputFile',
                         dest='inputFile',
                         help='file to mine. e.g. data.dat',
                         default='homework1.dat')
    optparser.add_option('-o', '--outputFile',
                         dest='outputFile',
                         help='output. e.g. out.txt',
                         default='result.txt')

    # get parameters
    (options, args) = optparser.parse_args()
    minsup = options.minS
    minconf = options.minC
    dataset = loadData(options.inputFile)
    resultfile = options.outputFile

    # mine frequent itemset and association rules
    start = time.clock()
    ap = Apriori(dataset, minsup, minconf)
    ap.mine()
    ap.print_to_file(resultfile)
    end = time.clock()

    print('mining completed.')
    print('min Support:' + str(minsup) + ', min Confidence:' + str(minconf) + '.')
    print('time cost: %.6fs.' % (end - start))

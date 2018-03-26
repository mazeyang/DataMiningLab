# DataMiningLab


The code attempts to implement Apriori algorithm.


List of files

1.	Project report (.pdf)
2.	Apriori.py
3.	homework1.dat (data file)
4.	README (this file)
5.  test.py (to run)


Dependent packages:

from collections import defaultdict
from itertools import combinations
from optparse import OptionParser
import time


Usage

Before run the program, you should install python 3.x.

To run the program with dataset provided and default values for minSupport = 0.8 and minConfidence = 0.9:
>> python test.py

To run the program with the dataset that you like and your minSupport and minConfidence:
>> python test.py -s minS -c minC -i inputFile -o outputFile


from collections import defaultdict
from itertools import combinations


class cached_property(object):

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls):
        if obj is None: return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


class Apriori:

    def __init__(self, transaction_list, minsup, minconf):

        self.transaction_list = list([frozenset(transaction) \
                                      for transaction in transaction_list])
        self.transaction_list_length = len(transaction_list)
        self.minsup = minsup
        self.minconf = minconf

        self.frequent_itemset = dict()
        self.frequent_itemset_support = defaultdict(float)
        self.maxitem = 1
        self.rule = []

    @cached_property
    def _get_items(self):
        items = set()
        for transaction in self.transaction_list:
            for item in transaction:
                items.add(item)
        return items

    def generate_frequent_itemset(self):

        def _get_next_candidate(itemset, length):
            # simply use F(k-1) x F(k-1) (itemset + itemset)
            return set([x.union(y) for x in itemset for y in itemset \
                        if len(x.union(y)) == length])

        def _filter_with_minsup(itemsets):
            local_counter = defaultdict(int)
            for itemset in itemsets:
                for transaction in self.transaction_list:
                    if itemset.issubset(transaction):
                        local_counter[itemset] += 1
            # filter with counter
            result = set()
            for itemset, count in local_counter.items():
                support = float(count) / self.transaction_list_length
                if support >= self.minsup:
                    result.add(itemset)
                    self.frequent_itemset_support[itemset] = support
            return result

        k = 1
        current_itemset = set()
        # generate 1-frequnt_itemset
        for item in self._get_items:
            current_itemset.add(frozenset([item]))
        self.frequent_itemset[1] = _filter_with_minsup(current_itemset)
        # generate k-frequent_itemset
        while True:
            k += 1
            current_itemset = _get_next_candidate(current_itemset, k)
            current_itemset = _filter_with_minsup(current_itemset)
            if current_itemset != set([]):
                self.frequent_itemset[k] = current_itemset
            else:
                self.maxitem = k - 1
                break
        return self.frequent_itemset

    def generate_rule(self):

        def _generate_rule(itemset, frequent_itemset_k):
            if len(itemset) < 2:
                return
            for element in combinations(list(itemset), 1):
                rule_head = itemset - frozenset(element)
                confidence = self.frequent_itemset_support[frequent_itemset_k] / \
                             self.frequent_itemset_support[rule_head]
                if confidence >= self.minconf:
                    rule = ((rule_head, itemset - rule_head), confidence)
                    # if rule not in self.rule, add and recall _generate_rule() in DFS
                    if rule not in self.rule:
                        self.rule.append(rule)
                        _generate_rule(rule_head, frequent_itemset_k)

        if len(self.frequent_itemset) == 0:
            self.generate_frequent_itemset()

        for key, val in self.frequent_itemset.items():
            if key == 1:
                continue
            for itemset in val:
                _generate_rule(itemset, itemset)
        return self.rule

    def mine(self):
        self.generate_frequent_itemset()
        self.generate_rule()

    def print_to_file(self, result_file='result.txt'):
        f = open(result_file, 'w')
        print('Frequent itemset:', file=f)
        for key, val in self.frequent_itemset.items():
            for itemset in val:
                if len(itemset) < self.maxitem:
                    continue
                print('(' + ', '.join(itemset) + ')', end='', file=f)
                print('   support = {0}'.format( \
                    round(self.frequent_itemset_support[itemset], 3)), file=f)
        print('\n========================================\n', file=f)
        print('Association rules:', file=f)
        for rule in self.rule:
            head = rule[0][0]
            tail = rule[0][1]
            if (len(head) + len(tail)) < self.maxitem:
                continue
            confidence = rule[1]
            print('(' + ', '.join(head) + ') ==> (' + ', '.join(tail) + ')', end='', file=f)
            print('  confidence = {0}'.format(round(confidence, 4)), file=f)
        f.close()

    def get_max_item(self):
        return self.maxitem
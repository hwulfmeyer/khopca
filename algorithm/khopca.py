import helper
import numpy


def clustering(data, kmin, kmax):
    if kmin >= kmax:
        print "Error: MIN must be smaller than MAX"
    else:
        # 1. create adjacency matrix
        data_adj = helper.create_Adjacent(data, 3)
        # 2. create value array for data
        data_value_array = numpy.zeros((data.shape[0],), dtype=numpy.int)
        # 2.5. fill value array with data

        # 3. apply all rules to data until nothing changes
        apply_rules_to_data(data_adj, data_value_array)
        # 4. output clustering result
        return get_data_labels


def get_data_labels(adjmatrix, data_array):
    cluster_labels = numpy.zeros((data_array.shape[0],), dtype=numpy.int)
    return cluster_labels


def apply_rules_to_data(adjmatrix, data_array):
    x = data_array.shape[0]
    for i in range(0,x,1):
        data_array[i] = rule_one(i, adjmatrix, data_array)
        data_array[i] = rule_two(i, adjmatrix, data_array)
        data_array[i] = rule_three(i, adjmatrix, data_array)
        data_array[i] = rule_four(i, adjmatrix, data_array)


def rule_one(nodeid, adjmatrix, data_array):
    """RULE 1
    if max(W(N(n))) > w_n
2     w_n = max(W(N(n))) - 1"""


def rule_two(nodeid, adjmatrix, data_array):
    """RULE 2
    if max(W(N(n)) == MIN & w_n == MIN
2     w_n = MAX;"""

def rule_three(nodeid, adjmatrix, data_array):
    """RULE 3
    if max(W(N(n))) <= w_n && w_n != MAX
2     w_n = w_n - 1;"""

def rule_four(nodeid, adjmatrix, data_array):
    """RULE 4
    if max(W(N(n)) == MAX && w_n == MAX
2     w_n = apply criterion to select a node from set (max(W(N(n)),w_n);
3     w_n = w_n - 1;"""
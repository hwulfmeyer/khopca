import helper
import numpy


def clustering(data, d="euclidean", knn=1, kmax=1):

    # data = data as numpy matrix
    # d = distance metric as string
    # kmax = maximum hops from the cluster center as integer
    # kNN = the k for k-neares-neighbors as integer

    kmin = 0
    data_length = data.shape[0]
    print "Clustering with: " + str(data_length) + "points, " + str(knn) + "=kNN, " + str(kmax) + "=kmax"

    if kmin >= kmax:
        print "Error: MIN must be smaller than MAX"
    elif data_length == 0:
        print "Error: No Data to cluster"
    else:
        # 1. create adjacency matrix
        data_adj = helper.create_Adjacent(data, knn)

        # 2. create value array for data
        data_value_array = numpy.zeros((data_length,), dtype=numpy.int)

        # 2.5. fill value array with data
        for i in range(0, data_length, 1):
            _, num = get_neighbors(i, data_adj, data_length)
            data_value_array[i] = kmin if num <= knn else kmax #TODO why value as kmin? randomn initilisation?

        # 3. apply all rules to data until nothing changes
        ## print data_value_array
        data_value_array = apply_rules_to_data(data_adj, data_value_array, data_length, kmin, kmax)
        print data_value_array
        # 4. output clustering result
        return get_data_labels(data_adj, data_value_array)

def get_neighbors(d, adjmatrix, length):
    neighbors = []
    num = 0

    for i in range(0,length,1):
        if adjmatrix[i][d] == 1:
            neighbors.append(i)
            num += 1

    return neighbors, num


def get_max_neighbor(nodeid, adjmatrix, data_array, length, kmin):
    cur_max = kmin
    neighbors, _ = get_neighbors(nodeid, adjmatrix, length)

    for i in range(0, len(neighbors), 1):
        if data_array[neighbors[i]] > cur_max:
            cur_max = data_array[neighbors[i]]

    return cur_max


def apply_rules_to_data(adjmatrix, data_array, length, kmin, kmax):
    something_changed = True

    while something_changed:
        something_changed = False

        for i in range(0, length, 1):
            cur_max = get_max_neighbor(i, adjmatrix, data_array, length, kmin)
            cur_node = data_array[i]

            data_array[i] = rule_one(data_array[i], cur_max)
            data_array[i] = rule_two(data_array[i], cur_max, kmin, kmax)
            data_array[i] = rule_three(data_array[i], cur_max, kmax)
            data_array[i] = rule_four(data_array[i], cur_max, kmax)

            if cur_node != data_array[i]:
                something_changed = True

    return data_array


def rule_one(cur_node, cur_max):
    """RULE 1
    if max(W(N(n))) > w_n
        w_n = max(W(N(n))) - 1"""

    if cur_max > cur_node:
        return cur_max-1
    else:
        return cur_node


def rule_two(cur_node, cur_max, kmin, kmax):
    """RULE 2
    if max(W(N(n)) == MIN & w_n == MIN
        w_n = MAX;"""

    if cur_max == kmin & cur_node == kmin:
        return kmax
    else:
        return cur_node


def rule_three(cur_node, cur_max, kmax):
    """RULE 3
    if max(W(N(n))) <= w_n && w_n != MAX
    w_n = w_n - 1;"""

    if cur_max <= cur_node & cur_node != kmax:
        return cur_node-1
    else:
        return cur_node


def rule_four(cur_node, cur_max, kmax):
    """RULE 4
    if max(W(N(n)) == MAX && w_n == MAX
        w_n = apply criterion to select a node from set (max(W(N(n)),w_n);
        w_n = w_n - 1;"""
    if cur_max == kmax & cur_node == kmax:
        ## criterion? random? more edges?
        return cur_node-1
    else:
        return cur_node


def get_data_labels(adjmatrix, data_array): #TODO labeling
    cluster_labels = numpy.zeros((data_array.shape[0],), dtype=numpy.int)
    return cluster_labels


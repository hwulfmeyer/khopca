import numpy
import random
from scipy.spatial import distance


# TODO: run by commandline

def cluster(data, knn, kmax, dstmeasure, subsampling=True):
    """Clusters data with the khopca algorithm

    Args:
        data: data as numpy matrix
        knn: the k for k-neares-neighbors as integer
        kmax: maximum hops from the cluster center as integer
        dstmeasure: distance metric as string

    Returns:
        A one-dimensional array containing the labels for data
        points, labels are integers

    Raises:
        none
    """
    subsamplesize = 0.75
    kmin = 0
    originaldata = data
    indices_kept = None

    print "Clustering with: " + str(data.shape[0]) + "points, " + str(knn) + "=kNN, " + str(kmax) + "=kmax"
    if subsampling:
        data, indices_kept = create_subsample(data, subsamplesize)
        print "using subsample of " + str(data.shape[0]) + " points for clustering"

    if kmin >= kmax:
        print "Error: MIN must be smaller than MAX"
    elif data.shape[0] == 0:
        print "Error: No Data to cluster"
    else:
        # 1. create adjacency matrix
        data_adj = create_adjacent(data, knn, dstmeasure)

        # 2. create value array for data
        data_value_array = numpy.zeros((data.shape[0],), dtype=numpy.int)

        # 2.5. fill value array with data
        for i in range(0, data.shape[0], 1):
            _, num = get_neighbors(i, data_adj)
            data_value_array[i] = num if num <= knn else kmax

        # 3. apply all rules to data until nothing changes
        data_value_array = apply_rules_to_data(data_adj, data_value_array, kmin, kmax)

        # 4. get labels for data
        data_labels = get_data_labels(data_adj, data_value_array, kmax)

        if subsampling:
            data_labels = fit_subsample_on_data(data, originaldata, data_labels, indices_kept)

        # 4. output clustering result
        return data_labels

def create_subsample(data, p):
    """Create a subsample of data with p(=percentege) of the data"""
    length = data.shape[0]
    keep = int(length * p)
    x = numpy.zeros((length,), dtype=numpy.int)

    for i in range(0, x.shape[0], 1):
        x[i] = i

    random.shuffle(x)
    x = x[:keep]
    newdata = numpy.zeros((keep, data.shape[1]), dtype=numpy.float)

    for i in range(0, x.shape[0], 1):
            newdata[i] = data[x[i]]

    return newdata, x

def create_adjacent(data, k, measure):
    '''

    :param data: given datapoints
    :param k: number of neighbors
    :param measure: euclidean
                    cityblock
                    seuclidean
                    sqeuclidean
                    cosine
                    correlation
                    hamming(matching)
                    jaccard
                    chebyshev
                    canberra
                    braycurtis
                    yule

    :return: adjacent matrix of points
    '''

    datapointcount = data.shape[0]   # number of datapoints
    # initialise datapointcount x datapointcount matrix with zeros
    adjacent = numpy.zeros([datapointcount, datapointcount], bool)
    print "using " + measure

    for row in range(0, datapointcount, 1):
        neighbor_points = []
        neighbor_dst = []

        for datapoint in range(0, datapointcount, 1):   # for every datapoint loop over every other datapoint

            if row == datapoint:
                continue

            pointvectors = (data[row], data[datapoint])

            dst = distance.pdist(pointvectors, measure)[0]

            if len(neighbor_points) < k:         # for the first k points they are our current neighbors
                neighbor_points.append(datapoint)
                neighbor_dst.append(dst)
            else:
                maxndst = max(neighbor_dst)
                if maxndst > dst:
                    # if found remove max-distance value and save new distance
                    index = neighbor_dst.index(maxndst)
                    del neighbor_dst[index]
                    neighbor_dst.append(dst)

                    del neighbor_points[index]       # save also the datapoint
                    neighbor_points.append(datapoint)

        for i in neighbor_points:    # construct adjacent matrix
            adjacent[row][i] = True
            adjacent[i][row] = True
    print "adjacentbuilding finished"
    return adjacent

def get_neighbors(nodeid, adjmatrix):
    neighbors = []
    num = 0

    for i in range(0, adjmatrix.shape[0], 1):
        if adjmatrix[i][nodeid]:
            neighbors.append(i)
            num += 1

    return neighbors, num

def get_max_neighbor(nodeid, adjmatrix, data_array, kmin):
    cur_max = kmin
    neighbors, _ = get_neighbors(nodeid, adjmatrix)

    for i in range(0, len(neighbors), 1):
        if data_array[neighbors[i]] > cur_max:
            cur_max = data_array[neighbors[i]]

    return cur_max

def apply_rules_to_data(adjmatrix, data_array, kmin, kmax):
    something_changed = True

    while something_changed:
        something_changed = False

        for i in range(0, data_array.shape[0], 1):
            cur_max = get_max_neighbor(i, adjmatrix, data_array, kmin)
            cur_node_old = data_array[i]

            if cur_max > data_array[i]:      # rule 1
                data_array[i] = cur_max - 1

            if cur_max == kmin and data_array[i] == kmin:      # rule 2
                data_array[i] = kmax

            if cur_max <= data_array[i] and data_array[i] != kmax:      # rule 3
                data_array[i] -= 1

            if cur_max == kmax and data_array[i] == kmax:      # rule 4
                # apply criterion to select a node from set (max(W(N(n)),w_n)
                data_array[i] -= 1

            if data_array[i] != cur_node_old:
                something_changed = True
    print "ruleapplying finished"
    return data_array

def get_data_labels(adjmatrix, data_array, kmax):
    cluster_labels = numpy.zeros((data_array.shape[0],), dtype=numpy.int)
    clustercenters = [i for i, elem in enumerate(data_array) if elem == kmax]
    print "found " + str(len(clustercenters)) + " cluster center"
    clusterid = 1
    for i in clustercenters:
        cluster_labels[i] = clusterid
        cur_neigh, _ = get_neighbors(i, adjmatrix)
        assign_clusters(cur_neigh, clusterid, kmax-1, adjmatrix, cluster_labels, data_array)
        clusterid += 1

    return cluster_labels

def assign_clusters(nodes, clusterid, cur_depth, adjmatrix, cluster_labels, data_array):
    if cur_depth >= 0:       # kmin = 0, stop if at edge of cluster
        for i in nodes:
            if cur_depth == data_array[i] and cluster_labels[i] == 0:
                cluster_labels[i] = clusterid
                cur_neigh, _ = get_neighbors(i, adjmatrix)
                assign_clusters(cur_neigh, clusterid, cur_depth-1, adjmatrix, cluster_labels, data_array)




def fit_subsample_on_data(data, originaldata, datalabels, indices):
    k = 1
    fitted_datalabels = numpy.zeros((originaldata.shape[0],), dtype=numpy.int)
    for ip in range(0,originaldata.shape[0],1):
        if ip in indices:
            fitted_datalabels[ip] = datalabels[numpy.where(indices == ip)]
            continue

        neighbor_points = []
        neighbor_dst = []

        for sp in range(0, data.shape[0], 1):
            dst = distance.euclidean(originaldata[ip], data[sp])
            if len(neighbor_points) < k:         # for the first k points they are our current neighbors
                neighbor_points.append(datalabels[sp])
                neighbor_dst.append(dst)
            else:
                maxndst = max(neighbor_dst)
                if maxndst > dst:
                    # if found remove max-distance value and save new distance
                    index = neighbor_dst.index(maxndst)
                    del neighbor_dst[index]
                    neighbor_dst.append(dst)

                    del neighbor_points[index]       # save also the datapoint
                    neighbor_points.append(datalabels[sp])
        fitted_datalabels[ip] = neighbor_points[0]

    return fitted_datalabels






import numpy
from scipy.spatial import distance





def cluster(data, knn, kmax, d):
    """Clusters data with the khopca algorithm

    Args:
        data: data as numpy matrix
        knn: the k for k-neares-neighbors as integer
        kmax: maximum hops from the cluster center as integer
        d: distance metric as string

    Returns:
        A one-dimensional array containing the labels for data
        points, labels are integers

    Raises:
        none
    """

    # TODO: subsampling of data -> time constraints for clustering

    kmin = 0
    data_length = data.shape[0]
    print "Clustering with: " + str(data_length) + "points, " + str(knn) + "=kNN, " + str(kmax) + "=kmax"

    if kmin >= kmax:
        print "Error: MIN must be smaller than MAX"
    elif data_length == 0:
        print "Error: No Data to cluster"
    else:
        # 1. create adjacency matrix
        data_adj = create_adjacent(data, knn)

        # 2. create value array for data
        data_value_array = numpy.zeros((data_length,), dtype=numpy.int)

        # 2.5. fill value array with data
        for i in range(0, data_length, 1):
            _, num = get_neighbors(i, data_adj)
            data_value_array[i] = num if num <= knn else kmax

        # 3. apply all rules to data until nothing changes
        data_value_array = apply_rules_to_data(data_adj, data_value_array, kmin, kmax)
        print data_value_array

        # 4. output clustering result
        return get_data_labels(data_adj, data_value_array, kmax)


def create_adjacent(data, k):
    datapointcount = data.shape[0]   # number of datapoints
    # initialise datapointcount x datapointcount matrix with zeros
    adjacent = numpy.zeros([datapointcount, datapointcount], bool)

    for row in range(0, datapointcount, 1):
        pointvector = data[row]
        neighbor_points = []
        neighbor_dst = []

        for datapoint in range(0, datapointcount, 1):   # for every datapoint loop over every other datapoint
            if row == datapoint:
                continue

            dst = distance.euclidean(pointvector, data[datapoint])

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

            if cur_max == kmin & data_array[i] == kmin:      # rule 2
                data_array[i] = kmax

            if cur_max <= data_array[i] & data_array[i] != kmax:      # rule 3
                data_array[i] -= 1

            if cur_max == kmax & data_array[i] == kmax:      # rule 4
                # apply criterion to select a node from set (max(W(N(n)),w_n)? random? more edges?
                data_array[i] -= 1

            if data_array[i] != cur_node_old:
                something_changed = True

    return data_array


def get_data_labels(adjmatrix, data_array, kmax):
    #TODO labeling
    cluster_labels = numpy.zeros((data_array.shape[0],), dtype=numpy.int)
    clustercenters = [i for i, elem in enumerate(data_array) if elem == kmax]

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
            if cur_depth == data_array[i]:
                cluster_labels[i] = clusterid
                cur_neigh, _ = get_neighbors(i, adjmatrix)
                assign_clusters(cur_neigh, clusterid, cur_depth-1, adjmatrix, cluster_labels, data_array)

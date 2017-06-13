import numpy
import random
from scipy.spatial import distance


def cluster(data, knn, kmax, dstmeasure, subsampling=True):
    """Clusters data with the khopca algorithm

    Args:
        data: data as numpy matrix
        knn: the k for k-neares-neighbors as integer
        kmax: maximum hops from the cluster center as integer
        dstmeasure: distance metric as string
        subsampling: if do subsample

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

    #print "Clustering with: " + str(data.shape[0]) + "points, " + str(knn) + "=kNN, " + str(kmax) + "=kmax"
    if subsampling:
        data, indices_kept = create_subsample(data, subsamplesize)
        #print "using subsample of " + str(data.shape[0]) + " points for clustering"

    if kmin >= kmax:
        print "Error: MIN must be smaller than MAX"
    elif data.shape[0] == 0:
        print "Error: No Data to cluster"
    else:
        # 1. create adjacency matrix
        #data_adj = create_adjacent(data, knn, dstmeasure)
        data_adj = get_approxiamte_knn_adjacency(data, knn)

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


class BinaryNode:
    def __init__(self, dim, index, median, points, parent, side):
        self.dimension = dim      # the dimension for which the median is for
        self.indexInData = index
        self.median = median      # the point containing the median for this node
        self.binpoints = points   # points contained in this node/bin
        self.parent = parent      # parentnode
        self.side = side        # side of this node left = True, right = False
        self.left = None    # "contains" points before median: < median
        self.right = None   # "contains" points after median: >= median

    @staticmethod
    def build_kdtree(points_in_bin, index_array, parent, side):
        # return None if empty
        if len(points_in_bin) == 0:
            return None
        else:
            # find dimension with greatest variance
            cur_greatest_dim = [0, 0]           # [value][dimension]
            for k in range(0, points_in_bin.shape[1], 1):
                dimvalues = points_in_bin[:, k]          # numpy array containing all values in column/dimension i
                # calculate variance & compare with current greatest
                curdim = numpy.var(dimvalues)
                if curdim >= cur_greatest_dim[0]:
                    cur_greatest_dim[0] = curdim
                    cur_greatest_dim[1] = k

            dimension = cur_greatest_dim[1]

            # find median
            sorted_points = points_in_bin[points_in_bin[:, dimension].argsort()]    # sort points according to dimension
            medianindex = int(points_in_bin.shape[0] / 2)  # int rounding = floor() => int(1/2) = 0
            median = numpy.copy(points_in_bin[medianindex])
            index_in_data = numpy.copy(index_array[medianindex])

            # create node for this iteration
            node = BinaryNode(dimension, index_in_data, median, points_in_bin, parent, "left")
            # Create child nodes and construct subtree with remaining binpoints
            # everythin from 0 to medianindex (excluding medianindex)
            node.left = BinaryNode.build_kdtree(
                points_in_bin[:medianindex],
                index_array[:medianindex],
                node,
                "left")
            # everythin medianindex to the end (exluding mdedianindex)
            node.right = BinaryNode.build_kdtree(
                points_in_bin[medianindex + 1:],
                index_array[medianindex + 1:],
                node,
                "right")
            return node

    @staticmethod
    def find_bin(node, q):

        if q[node.dimension] >= node.median[node.dimension]:    # if true => go right
            if node.right is not None:
                return node.find_bin(node.right, q)
            else:
                return node

        else:  # go left
            if node.left is not None:
                return node.find_bin(node.left, q)
            else:
                return node


def get_approxiamte_knn_adjacency(data, k):
    data_size = data.shape[0]   # number of datapoints
    adjacent = numpy.zeros([data_size, data_size], bool)
    index_array = numpy.zeros((data.shape[0],), dtype=numpy.int)
    for i in range(0,data.shape[0],1):
        index_array[i] = i
    data_kdtree = BinaryNode.build_kdtree(data, index_array, None, "")

    # iterate over all points and find the kNN in the kdtree
    for k in range(0, data_size, 1):
        neighbor_points = []
        neighbor_dst = []

        binofk = BinaryNode.find_bin(data_kdtree, data[k])
        # if knn = same point
        if (data[k]-binofk.median).all():
            binofk = binofk.parent

        neighbor_points.append(binofk.indexInData)
        neighbor_dst.append(distance.euclidean(data[k], binofk.median))

        found_knn = False
        side_to_add = None
        cur_node = binofk
        next_parent = None
        while not found_knn:
            next_parent = cur_node.parent

            # look if parent is smaller
            if next_parent is not None:
                maxndst = max(neighbor_dst)
                dst = distance.euclidean(data[k], next_parent.median)

                if len(neighbor_points) < k:  # for the first k points they are our current neighbors
                    neighbor_points.append(next_parent.indexInData)
                    neighbor_dst.append(dst)
                elif maxndst > dst:
                    # if found remove max-distance value and save new distance
                    index = neighbor_dst.index(maxndst)
                    del neighbor_dst[index]
                    neighbor_dst.append(dst)

                    del neighbor_points[index]  # save also the datapoint
                    neighbor_points.append(next_parent.indexInData)
                elif maxndst <= dst:
                    found_knn = True
            else:
                found_knn = True

            # traverse left/right node of parent of current one
            if cur_node.side == "Left":
                side_to_add = "Right"
            else:
                side_to_add = "Left"
            nodes = []
            if next_parent.right is not None and side_to_add == "Right":
                nodes.append(next_parent.right)
            if next_parent.left is not None and side_to_add == "Left":
                nodes.append(next_parent.left)
            while len(nodes) > 0:
                cur_while_node = nodes[-1]
                del nodes[-1]
                maxndst = max(neighbor_dst)
                dst = distance.euclidean(data[k], cur_while_node.median)

                if len(neighbor_points) < k:  # for the first k points they are our current neighbors
                    neighbor_points.append(cur_while_node.indexInData)
                    neighbor_dst.append(dst)
                elif maxndst > dst:
                    # if found remove max-distance value and save new distance
                    index = neighbor_dst.index(maxndst)
                    del neighbor_dst[index]
                    neighbor_dst.append(dst)

                    del neighbor_points[index]  # save also the datapoint
                    neighbor_points.append(cur_while_node.indexInData)

                #iterate over their childs
                if cur_while_node.right is not None:
                    nodes.append(cur_while_node.right)
                if cur_while_node.left is not None:
                    nodes.append(cur_while_node.left)
            cur_node = next_parent

        print neighbor_points
        for i in neighbor_points:    # construct adjacent matrix
            adjacent[k][i] = True
            adjacent[i][k] = True
    return adjacent


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
    """
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

    :return: adjacent matrix of points"""

    datapointcount = data.shape[0]   # number of datapoints
    # initialise datapointcount x datapointcount matrix with zeros
    adjacent = numpy.zeros([datapointcount, datapointcount], bool)

    for row in range(0, datapointcount, 1):
        neighbor_points = []
        neighbor_dst = []

        for datapoint in range(0, datapointcount, 1):   # for every datapoint loop over every other datapoint

            if row == datapoint:
                continue

            pointvectors = numpy.array([data[row], data[datapoint]])
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
    return data_array


def get_data_labels(adjmatrix, data_array, kmax):
    cluster_labels = numpy.zeros((data_array.shape[0],), dtype=numpy.int)
    clustercenters = [i for i, elem in enumerate(data_array) if elem == kmax]
    #print "found " + str(len(clustercenters)) + " cluster center"
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


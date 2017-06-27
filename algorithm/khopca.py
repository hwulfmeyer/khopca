import time
import numpy
import sys
from scipy.spatial import distance
from scipy.io import arff
from os import listdir
from os.path import isfile, join


def cluster(data, knn, kmax):
    """Clusters data with the khopca algorithm

    Args:
        data: data as numpy matrix
        knn: the k for k-neares-neighbors as integer
        kmax: maximum hops from the cluster center as integer

    Returns:
        A one-dimensional array containing the labels for data
        points, labels are integers
    """

    kmin = 0
    if kmin >= kmax:
        print "Error: MIN must be smaller than MAX"
        return None
    elif data.shape[0] == 0:
        print "Error: No Data to cluster"
        return None
    else:
        # 1. create adjacency matrix using best bin first
        data_adj = get_approxiamte_knn_adjacency(data, knn)

        # 2. create value array for data
        data_value_array = numpy.zeros((data.shape[0],), dtype=numpy.int)

        # 2.5. fill value array with data
        for i in range(0, data.shape[0], 1):
            _, num = get_neighbors(i, data_adj)
            data_value_array[i] = num if num <= knn else kmax

        # 3. apply all rules to data until nothing changes
        data_value_array = apply_rules_to_data(data_adj, data_value_array, kmin, kmax)

        # 4. get labels for data & output
        data_labels = get_data_labels(data_adj, data_value_array, kmax)
        return data_labels


class BinaryNode:
    def __init__(self, dim, median, points, parent, side):
        self.dimension = dim      # the dimension for which the median is for
        self.median = median      # the point containing the median for this node
        self.binpoints = points   # points contained in this node/bin with indices as last column
        self.parent = parent      # parentnode
        self.side = side          # side of this node left = L, right = R
        self.left = None          # "contains" points before median: < median
        self.right = None         # "contains" points after median: >= median

    @staticmethod
    def build_kdtree(data):
        data_size = data.shape[0]  # number of datapoints
        index_array = numpy.zeros(shape=(data_size,1), dtype=numpy.int)
        for i in range(0, data_size, 1):
            index_array[i][0] = i

        # append indexarray to datapoints
        binpoints = numpy.append(data, index_array, axis=1)
        return BinaryNode.__kdtree(binpoints, None, "")

    @staticmethod
    def __kdtree(points_in_bin, parent, side):
        # return None if empty
        if len(points_in_bin) == 0:
            return None
        else:
            # find dimension with greatest variance
            cur_greatest_dim = [0, 0]           # [value][dimension]
            for k in range(0, points_in_bin.shape[1]-1, 1):
                dimvalues = points_in_bin[:, k]          # numpy array containing all values in column/dimension i
                # calculate variance & compare with current greatest
                curdim = numpy.var(dimvalues)
                if curdim >= cur_greatest_dim[0]:
                    cur_greatest_dim[0] = curdim
                    cur_greatest_dim[1] = k
            dimension = cur_greatest_dim[1]

            # find median
            sorted_points = points_in_bin[points_in_bin[:, dimension].argsort()]    # sort points according to dimension
            medianindex = int(sorted_points.shape[0] / 2)  # int rounding = floor() => int(1/2) = 0
            median = numpy.copy(sorted_points[medianindex])

            # create node for this iteration
            node = BinaryNode(dimension, median, sorted_points, parent, side)
            # Create child nodes and construct subtree with remaining binpoints
            # everythin from 0 to medianindex (excluding medianindex)
            node.left = BinaryNode.__kdtree(sorted_points[:medianindex], node, "L")
            # everythin medianindex to the end (exluding mdedianindex)
            node.right = BinaryNode.__kdtree(sorted_points[medianindex + 1:], node, "R")
            return node

    @staticmethod
    def find_bin(node, q):
        if q[node.dimension] >= node.median[node.dimension]:    # if true => go right
            if node.right is not None:
                return node.find_bin(node.right, q)
            else:
                if node.left is not None:
                    return node.find_bin(node.left, q)
                else:
                    return node

        else:  # go left
            if node.left is not None:
                return node.find_bin(node.left, q)
            else:
                if node.right is not None:
                    return node.find_bin(node.right, q)
                else:
                    return node


def get_approxiamte_knn_adjacency(data, knn):
    data_size = data.shape[0]  # number of datapoints
    adjacent = numpy.zeros([data_size, data_size], bool)

    data_kdtree = BinaryNode.build_kdtree(data)

    # iterate over all points and find the kNN in the kdtree
    for k in range(0, data_size, 1):
        cur_datap = data[k]
        neighbor_points = []
        neighbor_dst = []

        binofk = BinaryNode.find_bin(data_kdtree, cur_datap)
        # if nearest point = input point dont append into neighbors and instead append parent
        if k == binofk.median[-1]:
            neighbor_points.append(binofk.parent.median[-1])   # append index of median in data
            neighbor_dst.append(distance.euclidean(cur_datap, binofk.parent.median[:-1]))  # append distance of median
        else:
            neighbor_points.append(binofk.median[-1])   # append index of median in data
            neighbor_dst.append(distance.euclidean(cur_datap, binofk.median[:-1]))  # append distance of median

        # look for our knn in the rest of the kdtree with a backtracing search towards the root
        cur_node = binofk
        while True:
            next_parent = cur_node.parent

            # look into parent if in knn
            if next_parent is not None:
                if next_parent.median[-1] != k:  # check if parent is our point

                    maxndst = max(neighbor_dst)
                    dst = distance.euclidean(cur_datap, next_parent.median[:-1])

                    if len(neighbor_points) < knn:  # for the first k points they are our current neighbors
                        neighbor_points.append(next_parent.median[-1])
                        neighbor_dst.append(dst)

                    elif maxndst > dst:
                        # if found nearer point remove max-distance value and save new distance
                        index = neighbor_dst.index(maxndst)
                        del neighbor_dst[index]
                        neighbor_dst.append(dst)

                        del neighbor_points[index]  # save also the datapoint
                        neighbor_points.append(next_parent.median[-1])
                    else:
                        break   # stop while loop if parent is larger than anything we found until now
            else:
                break   # stop while loop if at root

            # traverse childs of parent on other side
            datapoints = None
            if cur_node.side == "L" and next_parent.right is not None:
                datapoints = next_parent.right.binpoints
            elif cur_node.side == "R" and next_parent.left is not None:
                datapoints = next_parent.left.binpoints
            else:
                cur_node = next_parent
                continue

            # search in child points
            for dp in datapoints:  # for every datapoint loop over every other datapoint
                if dp[-1] != k:  # check if point is our point for knn
                    maxndst = max(neighbor_dst)
                    dst = distance.euclidean(cur_datap, dp[:-1])
                    if len(neighbor_points) < knn:  # for the first k points they are our current neighbors
                        neighbor_points.append(dp[-1])
                        neighbor_dst.append(dst)

                    elif maxndst > dst:
                        # if found nearer point remove max-distance value and save new distance
                        index = neighbor_dst.index(maxndst)
                        del neighbor_dst[index]
                        neighbor_dst.append(dst)

                        del neighbor_points[index]  # save also the datapoint
                        neighbor_points.append(dp[-1])

            cur_node = next_parent

        for i in neighbor_points:    # construct adjacent matrix
            adjacent[k][int(i)] = True
            adjacent[int(i)][k] = True
    return adjacent


def get_neighbors(nodeid, adjmatrix):
    """returns all the direct neighbors of nodeid in the graph

    Args:
        nodeid: index of the datapoint
        adjmatrix: adjmatrix with true=edge, false=noedge

    Returns:
        list of neighbors and the number of neighbors
    """
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
    """applies the khopca algorithm and all its rules to the data

    Args:
        adjmatrix: the adjmatrix with true=edge, false=noedge
        data_array: a 1d array with the khopca values of the data
        kmin: the kmin of the khopca algorithm
        kmax: the kmax of the khopca algorithm

    Returns:
        a 1d array with the khopca values of the data
    """
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
    """gets the labels for the data

    Args:
        adjmatrix: the adjmatrix with true=edge, false=noedge
        data_array: a 1d array with the khopca values of the data
        kmax: the kmax of the khopca algorithm

    Returns:
        numpy array of the datalabels
    """
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
            if cur_depth == data_array[i] and cluster_labels[i] == 0:
                cluster_labels[i] = clusterid
                cur_neigh, _ = get_neighbors(i, adjmatrix)
                assign_clusters(cur_neigh, clusterid, cur_depth-1, adjmatrix, cluster_labels, data_array)


def load_data(path):
    """
    Load a data set in arff format from path.

    :param path: to the data set
    :return: a numpy-matrix. column = attribute;  row a data point
    """
    data, meta = arff.loadarff(open(path, 'r'))
    if data.shape == (0,):
        return numpy.empty((0, len(meta._attributes)))
    else:
        data_transform = data[meta.names()[:-1]].copy().reshape(data.shape + (-1,))
        return data_transform.view(numpy.float)



def test_all_datasets(cluster_function):
    """
    Test cluster_function for all available data sets. The distance_measure parameter will always be measure.

    :param cluster_function: the clustering function. Signature is clustering(data: ndarray, distance_measure: str)
    :param measure: the distance measure to be used; as seen in scipys pdist
    :param process_count: process count

    :return: nothing
    """
    data_set_paths = [f for f in listdir("datasets") if isfile(join("datasets", f))]

    # function for verbosely testing the clustering function
    def cluster_with_catching(path):
        try:
            data = load_data("datasets/" + path)
            if data.shape[0] > 0:
                print str(path)
                start = time.time()
                cluster_function(data)
                m, s = divmod(time.time() - start, 60)
                print "Time:  " + str(int(m)) + "min " + str(int(s)) + "sec"
            else:
                print(str(path) + "is empty")
        except Exception as e:
            print(str(path) + " threw an EXCEPTION: " + str(type(e)) + ": " + str(e))

    for data_set_path in data_set_paths:
        cluster_with_catching(data_set_path)


if __name__ == "__main__":
    # konsolenprogramm
    '''
    newdata = load_data(str(sys.argv[1]))
    datalabels = cluster(newdata, int(sys.argv[2]), int(sys.argv[3]))
    if datalabels not None:
        for i in range(0, len(datalabels), 1):
            print datalabels[i]'''


    start = time.time()
    newdata = load_data("datasets/cf_TRex_test.arff")
    datalabels = cluster(newdata, 300, 9000)
    for i in range(0, len(datalabels), 1):
        pass
        #print datalabels[i]
    m, s = divmod(time.time()-start, 60)
    print "Time:  " + str(int(m)) + "min " + str(int(s)) + "sec"

    #test_all_datasets(lambda data: cluster(data, 300, 9000))

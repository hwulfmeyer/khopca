import time
import numpy
from scipy.io import arff
import khopca


def clustering_test(datapath, d, knn, kmax):
    data = load_data(datapath)
    print khopca.cluster(data, knn, kmax, d)


def load_data(path):
    """ Loads a data set from path. Data set must be arff format.

    Args:
        path: path to the data set

    Returns:
        a numpy-matrix. each column represents an attribute
        each row a data item
    """

    data, meta = arff.loadarff(open(path, 'r'))
    data_transform = data[meta.names()[:-1]].copy().reshape(data.shape + (-1, ))
    return data_transform.view(numpy.float)

if __name__ == "__main__":
    print "clustering start"
    start = time.time()
    clusterpath = "../iris_training.arff"
    #clusterpath = "../test.arff"
    #clusterpath = "../c_TRex.arff"
    #clusterpath = "../c_Aggregation.arff"
    clustering_test(clusterpath, "euclidean", 12, 20)
    m, s = divmod(time.time()-start, 60)
    print "Time:  " + str(int(m)) + "min " + str(int(s)) + "sec"
    print "clustering end"



import time
import numpy
from scipy.io import arff
import khopca
from os import listdir
from os.path import isfile, join
import sys



__available_measures = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine',
                        'euclidean', 'hamming', 'jaccard', 'matching',
                        'seuclidean', 'sqeuclidean',
                        'yule']

def clustering_test(datapath, dstmeasure, knn, kmax):
    data,_ = load_data(datapath)
    labels = khopca.cluster(data, knn, kmax, dstmeasure)
    for i in range(0,len(labels),1):
        print labels[i]

def load_data(path):
    """
       Load a data set in arff formatfrom path.

       :param path: to the data set
       :return: a numpy-matrix. column = attribute;  row a data point
       """
    data, meta = arff.loadarff(open(path, 'r'))
    if data.shape == (0,):
        return numpy.empty((0, len(meta._attributes))), 0
    else:
        data_matrix = numpy.zeros(shape=(data.shape[0], len(data[0]) - 1))

        for i in range(len(data)):
            arff_row = data[i]

            for j in range(len(arff_row) - 1):
                data_matrix[i][j] = arff_row[j]

    return data_matrix, data.shape[0]




def test_all_datasets(cluster_function, measure):
    """
    Test cluster_function for all available data sets. The distance_measure parameter will always be measure.

    :param cluster_function: the clustering function. Signature is clustering(data: ndarray, distance_measure: str)
    :param measure: the distance measure to be used; as seen in scipys pdist
    :param process_count: process count

    :return: nothing
    """
    print("test all sets with " + measure)
    data_set_paths = [f for f in listdir("datasets") if isfile(join("datasets", f))]

    # function for verbosely testing the clustering function
    def cluster_with_catching(path):
        try:
            data, length = load_data("datasets/" + path)
            if length > 0:
                print str(path)
                start = time.time()
                cluster_function(data, measure)
                m, s = divmod(time.time() - start, 60)
                print "Time:  " + str(int(m)) + "min " + str(int(s)) + "sec"
            else:
                print(str(path) + "is empty")
        except Exception as e:
            print(str(path) + " threw an EXCEPTION: " + str(type(e)) + ": " + str(e))

    for data_set_path in data_set_paths:
        cluster_with_catching(data_set_path)


def test_all_measures(cluster_function, data_set_path):
    """
    Test all available measures using one data set.

    :param cluster_function: the clustering function. Signature is clustering(data: ndarray, distance_measure: str)
    :param data_set_path: path to the data set
    :return: nothing
    """
    data, length = load_data(data_set_path)
    if length > 0:
        for measure in __available_measures:
            cluster_function(data, measure)


def run_tests(cluster_function):
    """
    Run all available tests

    :param cluster_function: the clustering function. Signature is clustering(data: ndarray, distance_measure: str),
        where distance_measure is a distance measurement function name as seen in scipys pdist
    :param process_count: number of parallel processes to use for testing. 1 will run sequentially.
    :return: nothing
    """
    for measure in __available_measures:
        test_all_datasets(cluster_function, measure)


if __name__ == "__main__":

    """print "clustering start"
    start = time.time()
    clusterpath = "../iris_training.arff"
    #clusterpath = "../test.arff"
    #clusterpath = "../c_TRex.arff"
    #clusterpath = "../c_Aggregation.arff"
    clustering_test(clusterpath, "euclidean", 35, 6000)
    m, s = divmod(time.time()-start, 60)
    print "Time:  " + str(int(m)) + "min " + str(int(s)) + "sec"
    print "clustering end"""

    clustering_test(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    #clustering_test("../test.arff", "euclidean" , 3, 5)


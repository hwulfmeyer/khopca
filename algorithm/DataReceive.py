import numpy

from scipy.io import arff
from scipy.spatial import distance


def __load_file(path):
    """
    Load a data set from path. Data set must be arff format.

    :param path: path to the data set
    :return: a numpy-matrix. each column represents an attribute; each row a data item
    """
    data, meta = arff.loadarff(open(path, 'r'))
    print("load " + path)

    return data[meta.names()[:-1]].view(numpy.float).reshape(data.shape + (-1,))

def test_on_data_set(clusterfunction, path):

    clusterfunction(__load_file(path))


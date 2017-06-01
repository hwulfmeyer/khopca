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


def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]


def create_Adjacent(data, k):

    datapointcount = data.shape[0]   #number of datapoints
    adjacent = numpy.zeros([datapointcount, datapointcount], float) #initialise datapointcount x datapointcount matrix with zeros


    for row in range(0,datapointcount,1):

        pointvector = data[row]
        neighborPoints = []
        neighborDst = []

        for datapoint in range(0,datapointcount,1):   #for every datapoint loop over every other datapoint

            if row == datapoint:
                continue

            dst = distance.euclidean(pointvector, data[datapoint])

            if len(neighborPoints) < k:     #at the beginning everything is your nearest neighbour
                neighborPoints.append(datapoint)
                neighborDst.append(dst)

            else:

                biggerIndexes = find_indices(neighborDst, lambda x: x > dst)  #search in neighborlist for old datapoints with greater distance

                if len(biggerIndexes) > 0:

                    index = neighborDst.index(max(neighborDst))         #if found remove max-distance value and save new distance
                    neighborDst.remove(neighborDst[index])
                    neighborDst.insert(index,dst)

                    neighborPoints.remove(neighborPoints[index])        #save also the datapoint
                    neighborPoints.insert(index, datapoint)

        for i in neighborPoints:                        # construct adjacent matrix
            adjacent[row][i] = 1
            adjacent[i][row] = 1

    return adjacent
















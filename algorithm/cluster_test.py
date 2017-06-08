import khopca
import helper
import time



def clustering_test(datapath, d, knn, kmax):
    data = helper.load_data(datapath)
    khopca.clustering(data, d, knn, kmax)


if __name__ == "__main__":
    print "clustering start"
    start = time.time()
    #clusterpath = "../iris_training.arff"
    #clusterpath = "../test.arff"
    clusterpath = "../c_TRex.arff"
    clustering_test(clusterpath, "euclidean", 50, 50)
    m, s = divmod(time.time()-start, 60)
    print "Time:" + str(int(m)) + "min  " + str(int(s)) + "sec"
    print "clustering end"



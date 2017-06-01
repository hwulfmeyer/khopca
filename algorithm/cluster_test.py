import khopca
import helper



def clustering_test(datapath, kmin, kmax):
    data = helper.load_data(datapath)
    print khopca.clustering(data, kmin, kmax)



if __name__ == "__main__":
    print "clustering start"
    clustering_test("../iris_training.arff", 0, 7)
    print "clustering end"


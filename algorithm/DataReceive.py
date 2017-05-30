import numpy as np
from scipy.spatial import distance

a = [1,2,3]
b = [1,2,4]
dst = distance.euclidean(a,b)
print(dst)
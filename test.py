from scipy.spatial import distance
import numpy


'''

def steuern(einkommen):
    """BLABLABLA TESTFUNKTION"""
    if einkommen <= 50:
        steuer = 0
    elif einkommen == 51:
        steuer = 1
    else:
        steuer = 2 * (einkommen-51)

    while steuer <= 10:
            steuer += 1

    return steuer



print(steuern(input("Steuern: ")))

def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]

a = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
a.sort()
print (a)
a = a[::-1]
print(a)

b = []
b.append(1)
b.append(2)
b.append(3)
b[2] = 5
print(b)

c = find_indices(b, lambda e: e > 4)
print c

adjacent = numpy.zeros([2,2], float)
print adjacent[1][1]

'''

x = numpy.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]])

print x

print x.shape

y = numpy.zeros((10,), dtype=numpy.int)

print y.shape[0]
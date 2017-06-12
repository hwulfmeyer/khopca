import numpy


x = numpy.array([[ 1,  55,  3],[ 4,  25,  6],[ 7,  8,  9],[10, 11, 12]])
print x[:,1]    # gebe spalte 1 aus
print x[0]      # gebe zeile 0 aus
print x[0][1]   # x[zeile][spalte]
print x[1:]


y = numpy.array([])
z = numpy.zeros((1,0))



print len(x)



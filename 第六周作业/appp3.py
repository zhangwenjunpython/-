import pandas
import numpy
data = pandas.read_csv('aapl.csv')
d=numpy.average(data['Volume'])
print(d)

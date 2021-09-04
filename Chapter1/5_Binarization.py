import numpy as np
from numpy.core.fromnumeric import take
from sklearn import preprocessing

data= np.array([[1, 2, 3, 4],[3, 1, 4, 2],[4, 3, 2, 1],[2, 4, 1, 3]])

data_binarized=preprocessing.Binarizer(threshold=2.5).transform(data)
print(data_binarized)
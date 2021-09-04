import numpy as np
from numpy.core.fromnumeric import take
from sklearn import preprocessing

data= np.array([[1, 2, 3, 4],[3, 1, 4, 2],[4, 3, 2, 1],[2, 4, 1, 3]])

data_normalized = preprocessing.normalize(data,norm='l1', axis=0)
print(data_normalized)

data_norm_abs=np.abs(data_normalized)
print(data_norm_abs)


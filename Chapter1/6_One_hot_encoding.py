import numpy as np
from numpy.core.fromnumeric import take
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

data = np.array([[1, 1, 2], [0, 1, 1], [1, 2, 3]])
print(data)

encoder = preprocessing.OneHotEncoder()
encoder.fit(data.astype(int))

encoded_vector = encoder.transform([[1, 2, 3]]).toarray()
print(encoded_vector)
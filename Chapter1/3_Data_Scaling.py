import numpy as np
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
data= np.array([[1, 2, 3, 4],[3, 1, 4, 2],[4, 3, 2, 1],[2, 4, 1, 3]])

print("Min :",data.min(axis=0))
print("Max :",data.max(axis=0))

data_scaler=preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled=data_scaler.fit_transform(data)
# X scaled = (X-Xmin)/(Xmax-Xmin)
print("Min: ",data_scaled.min(axis=0))
print("Max: ",data_scaled.max(axis=0))

print(data_scaled)
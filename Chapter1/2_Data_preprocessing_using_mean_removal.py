import numpy as np
from numpy.core.fromnumeric import take
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

data= np.array([[1, 2, 3, 4],[5, 6, 7, 8],[0, 10, 11, 12],[13, 14, 15, 16]])
print("Mean: ",data.mean(axis=0))#in gia tri trung binh tren 1 row, 0 col

print("Standard Deviation: ",data.std(axis=0))#in ra do lech chuan tren 1 row, 0 col

data_standardized = preprocessing.scale(data)

print("Mean standardized data: ",data_standardized.mean(axis=0))
#in ra gia tri trung binh cua data duoc tieu chuan hoa sau khi tien xu ly

print("Standard Deviation standardized data:",data_standardized.std(axis=0))
#in ra do lech chuan cua data sau khi tien xu ly
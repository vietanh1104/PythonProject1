import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

X = np.array([ [1.2, 3] , [2, 3] , [3, 1] , [2.4, 7] , [5, 7] , [4,6] , [6,1] , [7,5] , [6.5,3 ] ])
y = np.array([0,0,0,1,1,1,2,2,2])

classifier =linear_model.LogisticRegression(solver="lbfgs",C=100)
classifier.fit(X,y)

x_max, x_min = max(X[:,0])+1 , min(X[:,0])-1
y_max, y_min = max(X[:,1])+1 , min(X[:,1])-1

x_values, y_values =np.meshgrid(np.arange(x_min,x_max,0.01),np.arange((y_min),y_max,0.01))
# np.meshgrid tao ra ma tran x_values vuong tu x_min den x_max voi step 0,01 axis=0
# np.meshgrid tao ra ma tran y_values vuong tu y_min den y_max voi step 0,01 axis=1

mesh_output = classifier.predict(np.c_[x_values.ravel(),y_values.ravel() ] )
mesh_output = mesh_output.reshape(x_values.shape)
#np.c_:Stack 1-D arrays as columns into a 2-D array.
#np.ravel() tao ma tran chuyen vi
plt.figure()
plt.pcolormesh(x_values,y_values ,mesh_output, cmap=plt.cm.gray)
# specify the boundaries of the figure
plt.xlim(x_values.min(), x_values.max())
plt.ylim(y_values.min(), y_values.max())
# specify the ticks on the X and Y axes
plt.scatter(X[:,0],X[:,1],c=y,s=80,edgecolors="blue")
# s:size
# edgecolor : vien
plt.xticks((np.arange(int(min(X[:, 0])-1), int(max(X[:, 0])+1),1.0)))
plt.yticks((np.arange(int(min(X[:, 1])-1), int(max(X[:, 1])+1),1.0)))
plt.show()
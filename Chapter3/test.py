import numpy as np
x = np.array([[0,1],[1,0],[2,1],[3,0]])
print x
class_0 = np.array([x[i,:] for i in range(len(x)) if x[i,1]==1])
print class_0
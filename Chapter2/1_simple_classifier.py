import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1,2],[1,4],[3,4],[2,3],[4,5],[6,4],[6,2],[8,3]])
y = [1,0,0,1,1,0,1,0]

class_0 = np.array([ X[i] for i in range( len(X) ) if y[i]==0])
class_1 = np.array([ X[i] for i in range( len(X) ) if y[i]==1])

line_x=range(10)
line_y=line_x

plt.figure()
plt.scatter( class_0[:,0] , class_0[:,1], color='black',marker='s')
plt.scatter( class_1[:,0] , class_1[:,1],color='black', marker='x')
plt.plot(line_x,line_y,color='blue',linewidth=3)
plt.show()
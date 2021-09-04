import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
input_file = 'data_multivar.txt'
X = []
y = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        X.append(data[:-1])
        y.append(data[-1])
X = np.array(X)
y = np.array(y)

classifier_guassiannb = GaussianNB()
classifier_guassiannb.fit(X,y)
y_pred = classifier_guassiannb.predict(X)

accuracy = 100.0 * ( y == y_pred ).sum() / X.shape[0]
print(' Accuracy of the classifier = ', accuracy,' %')

x_max, x_min = max(X[:,0]) + 1.0 , min(X[:,0])-  1.0
y_max, y_min = max(X[:,1]) + 1.0 , min(X[:,1]) - 1.0

# denotes the step size that will be used in the mesh grid
step_size = 0.01
# define the mesh grid
x_values, y_values = np.meshgrid(np.arange(x_min, x_max,step_size), np.arange(y_min, y_max, step_size))

mesh_output = classifier_guassiannb.predict( np.c_[ x_values.ravel(),y_values.ravel()])
mesh_output = mesh_output.reshape(x_values.shape)

plt.figure()

plt.pcolormesh(x_values,y_values,mesh_output,cmap=plt.cm.gray)
plt.scatter( X[:,0] , X[:,1] , c=y , s=15 , edgecolors='black')

plt.xlim( x_values.min() , x_values.max())
plt.ylim( y_values.min() , y_values.max())

plt.xticks( np.arange( int( min(X[:,0]) - 1), int( max(X[:,0]) + 1 )))
plt.yticks( np.arange( int( min(X[:,1]) - 1), int( max(X[:,1]) + 1 )))
plt.show()
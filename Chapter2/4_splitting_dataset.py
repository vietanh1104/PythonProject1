import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
import matplotlib.pyplot as plt

input_file = 'data_multivar.txt'
X = []
y = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data=[float(x) for x in line.split(',')]
        X.append(data[:-1])
        y.append(data[-1])
X=np.array(X)
y=np.array(y)
print X
print y

#splitting the dataset for training and testing
X_train, X_test, y_train, y_test= model_selection.train_test_split(X,y,test_size=0.25,random_state=5)

classifier_gaussiannb_new = GaussianNB()
classifier_gaussiannb_new.fit(X_train,y_train)

y_test_pred = classifier_gaussiannb_new.predict(X_test)

accuracy = 100.0 * ( y_test == y_test_pred).sum() / X_test.shape[0]
print('Accuracy of the classifier = ',round(accuracy,2),' %')

X = X_test
y = y_test

x_min, x_max = min(X[:,0]) - 1, max(X[:,0]) + 1
y_min, y_max = min(X[:,1]) - 1, max(X[:,1]) + 1

x_values , y_values = np.meshgrid( np.arange(x_min,x_max,0.01), np.arange(y_min,y_max,0.01))

mesh_output = classifier_gaussiannb_new.predict( np.c_[x_values.ravel(),y_values.ravel()])
mesh_output = mesh_output.reshape(x_values.shape)

plt.figure()
plt.pcolormesh( x_values, y_values, mesh_output, cmap=plt.cm.gray)
plt.scatter( X[:,0], X[:,1],c=y, s=15, edgecolor='green', linewidths= 1, cmap=plt.cm.Paired )
plt.xticks(np.arange(int(min(X[:,0])-1),int(max(X[:,0])+1),1.0))
plt.yticks(np.arange(int(min(X[:,1])-1),int(max(X[:,1])+1),1.0))
plt.xlim(x_values.min(),x_values.max())
plt.ylim(y_values.min(),y_values.max())
plt.show()
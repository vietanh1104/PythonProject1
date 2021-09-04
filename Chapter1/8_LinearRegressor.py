import numpy as np
from numpy.core.fromnumeric import take
from sklearn import preprocessing

filename = "Dataset.txt"
X = []
y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        X.append(xt)
        y.append(yt)

num_training = int(0.8 * len(X))#use 80% DATASET for training
num_test = len(X) - num_training#20% Dataset for test

# Training data
X_train = np.array(X[:num_training]).reshape((num_training,1))
y_train = np.array(y[:num_training])
# Test data
X_test = np.array(X[num_training:]).reshape((num_test,1))
y_test = np.array(y[num_training:])

from sklearn import linear_model
# Create linear regression object
linear_regressor = linear_model.LinearRegression()
# Train the model using the training sets
linear_regressor.fit(X_train, y_train)

y_train_pred = linear_regressor.predict(X_train)

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, y_train_pred, color='black', linewidth=3.5)
plt.title('Training data')
plt.show()

y_test_pred = linear_regressor.predict(X_test)

plt.figure()
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_test_pred, color='black', linewidth=3.5)
plt.title('Test data')
plt.show()
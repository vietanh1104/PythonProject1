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

y_test_pred = linear_regressor.predict(X_test)

import sklearn.metrics as sm
import pickle

output_model_file = "3_model_linear_regr.pkl"

with open(output_model_file, 'wb') as f:
    pickle.dump(linear_regressor, f)
with open(output_model_file, 'rb') as f:
    mode_linregr = pickle.load(f)

y_test_pred_new = mode_linregr.predict(X_test)

print("New mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))
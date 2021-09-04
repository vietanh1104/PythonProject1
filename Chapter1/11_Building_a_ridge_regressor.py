import numpy as np
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
import sklearn.metrics as sm

ridge_regressor = linear_model.Ridge(alpha=0.01, fit_intercept=True, max_iter=10000)
ridge_regressor.fit(X_train, y_train)

y_test_pred_ridge= ridge_regressor.predict(X_test)

print( "Mean absolute error =", round( sm.mean_absolute_error(y_test, y_test_pred_ridge), 2))
print( "Mean squared error =", round( sm.mean_squared_error(y_test,y_test_pred_ridge), 2))
print( "Median absolute error =", round( sm.median_absolute_error(y_test, y_test_pred_ridge), 2))
print( "Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred_ridge), 2))
print( "R2 score =", round( sm.r2_score(y_test, y_test_pred_ridge), 2))

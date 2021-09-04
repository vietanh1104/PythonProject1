import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve
X = []
input_file = 'car.txt'
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = line[:-1].split(',')
        X.append(data)
X = np.array(X)

label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    label_encoder.append(preprocessing.LabelEncoder())
    X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

classifier = RandomForestClassifier(random_state=7)
parameter_grid = np.array([200,500,800,1100])
train_scores, validation_scores = validation_curve(classifier, X, y, "n_estimators", parameter_grid, cv=5)
print('\n ##### LEARNING CURVES #####')
print("\n Training scores: \n", train_scores)
print("\n Validation scores: \n", validation_scores)
# Plot the curve

plt.figure()
plt.plot(parameter_grid, 100*np.average(train_scores, axis=1), color='black')
plt.title('Learning curve')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.show()
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection

X = []
y = []
input_file = 'wine.data.txt'
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        X.append(data[1:])
        y.append(data[0])
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)
classifier_DecisionTree = DecisionTreeClassifier()
classifier_DecisionTree.fit(X_train, y_train)

y_test_pred = classifier_DecisionTree.predict(X_test)

accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print 'Accuracy of the classifier = ',round(accuracy,2),'%'
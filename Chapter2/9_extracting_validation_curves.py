import numpy as np
X = []
input_file = 'car.txt'
with open(input_file,'r') as f:
    for line in f.readlines():
        data = line[:-1].split(',')
        X.append(data)
X = np.array(X)
#Convert string into numberic data
from sklearn import preprocessing
label_encoder = []
X_encoded =  np.empty(X.shape)
for i, item in enumerate(X[0]):
    label_encoder.append(preprocessing.LabelEncoder())
    X_encoded[: ,i] = label_encoder[-1].fit_transform(X[:,i])
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve
#  Change n_entimators parameter
classifier = RandomForestClassifier( max_depth=4, random_state=7)
parameter_grid = np.linspace( 25, 200, 8).astype(int)

train_score, validation_score = validation_curve(classifier, X, y, 'n_estimators',parameter_grid,cv=5)
print ('##########Validaion curve###########')
print ('Param: n_estimators\nTraining score:\n',train_score)
print ('Param: n_estimators\nValidation score\n',validation_score)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(parameter_grid,100*np.average(train_score,axis=1),color='gray')
plt.xlabel('Number of estimators')
plt.ylabel('Accuracy')
plt.title('Training curve')
plt.show()
# Change max_depth parameter
classifier = RandomForestClassifier( n_estimators=20, random_state=7)
parameter_grid = np.linspace(2,10,5).astype(int)

train_score, validation_score = validation_curve(classifier, X, y, 'max_depth', parameter_grid, cv=5)
print "Param: max_depth\nTraining score\n",train_score
print "Param: max_depth\nValidation score\n",validation_score

plt.figure()
plt.plot( parameter_grid, 100*np.average(train_score,axis=1), color='gray')
plt.ylabel('Accuracy')
plt.xlabel('Max depth of the tree')
plt.title('Validation curve')
plt.show()
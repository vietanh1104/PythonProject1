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
#Building a Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
params = { 'n_estimators' : 200, 'max_depth': 8 , 'random_state': 7 }
classifier = RandomForestClassifier(**params)
classifier.fit(X,y)
#Cross_validation
from sklearn import model_selection
accuracy = model_selection.cross_val_score( classifier, X, y,scoring='accuracy', cv=3)
print 'Accuracy: '+str( round( 100 * accuracy.mean(), 2 ))+' %'
#Testing
input_data = ['vhigh' , 'med' , '2' , '4' , 'big' , 'high' ]
input_data_encoded = [0] * len(input_data)
for i, item in enumerate(input_data):
    input_data_encoded[i] = int(label_encoder[i].transform([input_data[i]]))
input_data_encoded=np.array(input_data_encoded)
#Predict the test
output_class =  classifier.predict([input_data_encoded])
print'Out class: ', label_encoder[-1].inverse_transform(output_class)[0]
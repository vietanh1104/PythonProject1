import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection

X=[]
y=[]
input_file= 'data_multivar.txt'
with open(input_file,'r') as f:
    for line in f.readlines():
        data=[float(x) for x in line.split(',')]
        X.append(data[:-1])
        y.append(data[-1])
X = np.array(X)
y = np.array(y)

classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X,y)

num_validation = 5
accuracy = model_selection.cross_val_score(classifier_gaussiannb,X,y,
                                            scoring='accuracy',cv=num_validation)
print "Accuracy: "+ str(round(100*accuracy.mean(),2 ))+  "%"

f1 = model_selection.cross_val_score(classifier_gaussiannb,X,y,
                                            scoring='f1_weighted',cv=num_validation)
print "F1 score: "+ str(round(100*f1.mean(),2))+ "%"

precision = model_selection.cross_val_score(classifier_gaussiannb,X,y,
                                            scoring='precision_weighted',cv=num_validation)
print 'Precision: '+str(round(100*precision.mean(),2))+"%"

recall =  model_selection.cross_val_score(classifier_gaussiannb,X,y,
                                            scoring='recall_weighted',cv=num_validation)
print 'Recall: '+str(round(100*recall.mean(),2))+"%"
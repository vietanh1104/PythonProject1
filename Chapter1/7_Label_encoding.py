import numpy as np
from numpy.core.fromnumeric import take
from sklearn import preprocessing

input_classes=[ ' hanoi ',' hanam ',' hatay ',' habac ',' hadong ']

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(input_classes)

print("Class mapping: ")
for i, item in enumerate(label_encoder.classes_):
    print(item, "-->", i)

labels =  [' hanoi ',' hadong ',' hanam ']
encoded_labels = label_encoder.transform(labels)

print("Labels =",labels)
print("List encoded labels=", list(encoded_labels) )

encoded_labels = [1, 2, 0, 4]
decoded_labels = label_encoder.inverse_transform(encoded_labels)

print("Encoded labels =", encoded_labels)
print("Decoded labels =", list(decoded_labels))
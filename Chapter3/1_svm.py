import numpy as np
X = []
y = []
input_file = 'data_multivar.txt'
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = line[:-1].split(',')
        X.append(data[:-1])
        y.append(data[-1])
X = np.array(X)
y = np.array(y)

class_0 = np.array([X[i] for i in range(len(X)) if y[i] == "0"])
class_1 = np.array([X[i] for i in range(len(X)) if y[i] == "1"])

import numpy as np
import csv

n_cluster=3
arr=[]
filename="Iris.csv"
file_reader = csv.reader(open(filename, 'r'), delimiter=',')
for row in file_reader:
  arr.append(row[1:5])

arr=np.array(arr[1:]).astype(np.float)


def InitKmeans(n_cluster, arr=None):
    return arr[np.random.choice(arr.shape[0], n_cluster, replace=False)]


def Distance(vectorA, vectorB):
    return np.sum(np.sqrt((vectorA - vectorB) ** 2))

def Clusting(X,centroid ):
    store=[ [] for i in range(len(centroid))]
    for i in range(len(X)):
        min = 0x3f3f3f3f3f3f3f
        pos =-1
        for j in range(len(centroid)):
            d = Distance(X[i],centroid[j])
            if d<min:
                min=d
                pos=j
        store[pos].append(X[i])
    return np.array(store)


def Update(store):
    newMeans=[[] for i in range(len(store))]
    for i in range(len(store)):
        newMeans[i] = np.average(store[i], 0)
    return newMeans


def fit(n_cluster,X,epouch):
    newCentroid=InitKmeans(n_cluster,X)
    for i in range(epouch):
        store=Clusting(X,newCentroid)
        newCentroid=Update(store)
    return newCentroid

newcentroid=np.array(fit(3,arr,3))

res=[]
for i in range(4):
    x=float(input())
    res.append(x)
res = np.array(res)


type_of_flower=(["Iris-setosa","Iris-versicolor","Iris-virginica"])
pos = -1
Distance_min = 0x3f3f3f3f3f

for i in range(3):
    d=Distance(res,newcentroid[i])
    if d<Distance_min:
        Distance_min=d
        pos=i

print(type_of_flower[pos])

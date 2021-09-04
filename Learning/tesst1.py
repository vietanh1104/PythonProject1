import numpy as np
import csv
import warnings
warnings.filterwarnings("ignore")
arr=[]
filename="Iris.csv"
file_reader = csv.reader(open(filename, 'r'), delimiter=',')
for row in file_reader:
  arr.append(row[1:5])
arr=np.array(arr[1:]).astype(np.double)


def InitKmeans(n_clus=3, X=arr):
  amax = np.max(X, 1)
  amin = np.min(X, 1)
  means = np.zeros((n_clus, X.shape[1]))
  for i in range(n_clus):
    for j in range(X.shape[-1]):
      means[i][j] = np.random.uniform(amin[j], amax[j])
  return means


def Distance(vectorA, vectorB):
  return np.sum(np.sqrt((vectorA - vectorB) ** 2))


def Clusting(X, means):
  store = [ [] for i in range(len(means)) ]
  for i in range(len(X)):
    min = 0x3f3f3f3f3f3f
    index = -1
    for j in range(len(means)):
      d = Distance(X[i], means[j])
      if d < min:
        min = d
        index = j
    store[index].append(X[i])
  return np.array(store)


def Update(store):
  newMeans = [[] for i in range(len(store))]
  for i in range(len(store)):
    newMeans[i] = np.average(store[i], 0)
  return newMeans


def fit(n_clus, X, epouch):
  means = InitKmeans(n_clus, X)

  for i in range(epouch):
    store = Clusting(X, means)
    means = Update(store)

  return means


def Distance(vecto1,vecto2):
  return np.sum(np.sqrt((vecto1-vecto2)**2))
def Clusting(X, means):
  store = [[] for i in range(len(means))]
  for i in range(len(X)):
    min = 0x3f3f3f3f3f3f3f3f
    index = -1
    for j in range(len(means)):
      d = Distance(X[i], means[j])
      if d < min:
        min = d
        index = j
    store[index].append(X[i])
  return store


def Update(store):
  newMeans = [[] for i in range(len(store))]
  for i in range(len(store)):
    newMeans[i] = np.average(np.array(store[i]), 0)
  return newMeans


def fit(nClus, X, epouch):
  means = InitKmeans(nClus, X)

  for i in range(epouch):
    store = Clusting(X, means)
    means = Update(store)

  return means
my_kmeans= fit(3,arr,50)

from sklearn.cluster import KMeans
clusting=KMeans(3)
res=clusting.fit(arr).cluster_centers_

for i in range(3):
  print( np.sum(np.sqrt((res[i]-my_kmeans[i])*2)))

